import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from models import CLASSES, SIZE, build, transform, head_and_features
from fruits import FRUITS, paths

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'artifacts'
EXPERIMENTS = [
    dict(id='baseline', name='Small CNN', architecture='cnn', lr=0.001, transfer=False,
         changed='Baseline: small CNN, learning rate 0.001', why='Establish performance when learning visual features from scratch.'),
    dict(id='cnn_low_lr', name='CNN · lower LR', architecture='cnn', lr=0.0001, transfer=False,
         changed='Only learning rate: 0.001 → 0.0001', why='Measure sensitivity to smaller optimization steps at the same training budget.'),
    dict(id='resnet18', name='ResNet18 · transfer', architecture='resnet18', lr=0.001, transfer=True,
         changed='Small CNN → ImageNet ResNet18 with frozen backbone and trainable linear head', why='Test whether pretrained residual features improve generalization. Architecture and initialization change together.'),
    dict(id='mobilenet', name='MobileNetV3 · transfer', architecture='mobilenet_v3_small', lr=0.001, transfer=True,
         changed='Small CNN → ImageNet MobileNetV3 Small with frozen backbone and trainable classifier', why='Evaluate a compact transfer model. Compare with ResNet18 under the same frozen-backbone protocol.'),
]

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class Images(Dataset):
    def __init__(self, frame, augment=False):
        self.rows = frame.reset_index(drop=True)
        self.tf = transform(augment)
        self.images = []
        for p in self.rows.path:
            with Image.open(ROOT / p) as im:
                self.images.append(ImageOps.exif_transpose(im).convert('RGB').resize((SIZE, SIZE)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.tf(self.images[i]), CLASSES.index(self.rows.iloc[i].label)

def scores(y, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=range(len(CLASSES)), average='macro', zero_division=0)
    return dict(accuracy=float(accuracy_score(y, pred)), precision=float(p), recall=float(r), f1=float(f))

def pass_epoch(model, loader, criterion, optimizer=None):
    model.train(optimizer is not None)
    total, labels, guesses, grad_norms = 0., [], [], []
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss: training stopped')
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                grad_norms.append(float(norm))
                optimizer.step()
            total += float(loss.detach()) * len(y)
            labels.extend(y.tolist())
            guesses.extend(logits.argmax(1).tolist())
    return dict(loss=total / len(labels), **scores(labels, guesses),
                gradient_mean=float(np.mean(grad_norms)) if grad_norms else 0.,
                gradient_max=max(grad_norms, default=0.))

def diagnostics(history):
    h = pd.DataFrame(history)
    last = h.iloc[-1]
    gap = float(last.train_accuracy - last.val_accuracy)
    best_loss_index = h.val_loss.idxmin()
    loss_divergence = bool(last.val_loss > h.val_loss.min() * 1.15 and
                           last.train_loss < h.loc[best_loss_index, 'train_loss'])
    plateau = len(h) >= 4 and float(h.val_loss.tail(3).max() - h.val_loss.tail(3).min()) < .025
    return [
        dict(topic='Overfitting', flag=bool(gap > .10 or loss_divergence), detail=f'Last-epoch accuracy gap: {gap:.3f}. Flag if gap >0.10 OR validation loss rises >15% above its minimum while training loss decreases. Loss-divergence indicator: {loss_divergence}.'),
        dict(topic='Underfitting', flag=bool(last.train_accuracy < .70 and last.val_accuracy < .70), detail='Heuristic: both accuracies below 0.70. This may also indicate difficult data or insufficient epochs.'),
        dict(topic='Unstable training', flag=bool((h.val_loss.diff().abs() > .30).any()), detail='Flag if consecutive validation losses differ by more than 0.30.'),
        dict(topic='Convergence', flag=bool(plateau), detail='Plateau indicator: last 3 validation losses span < 0.025; a fixed budget does not prove convergence.'),
        dict(topic='Vanishing gradients', flag=bool((h.gradient_mean.tail(3) < 1e-6).all()), detail='Flag if mean trainable-parameter gradient norm stays below 1e-6 for last 3 epochs. Frozen backbones are excluded.'),
        dict(topic='Large / exploding gradients', flag=bool((h.gradient_max > 100).any()), detail=f'Max pre-clipping norm: {h.gradient_max.max():.3f}. Flag >100; clip threshold 5. Non-finite gradients abort training.'),
    ]

def curves(history, folder):
    h = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, metric in zip(axes, ['loss', 'accuracy', 'f1']):
        ax.plot(h.epoch, h['train_' + metric], label='Training')
        ax.plot(h.epoch, h['val_' + metric], label='Validation')
        ax.set(xlabel='Epoch', title=metric.replace('f1', 'Macro F1').title())
        ax.legend()
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(folder / 'curves.png', dpi=140)
    plt.close(fig)

def evaluate(model, dataset, frame, folder, batch):
    model.eval()
    probs = []
    start = time.perf_counter()
    with torch.inference_mode():
        for x, _ in DataLoader(dataset, batch_size=batch):
            probs.extend(model(x).softmax(1).tolist())
    ms = (time.perf_counter() - start) * 1000 / len(dataset)
    pred = np.argmax(probs, axis=1)
    y = [CLASSES.index(s) for s in frame.label]
    result = scores(y, pred)
    result['inference_ms_per_image_batched'] = ms
    result['confusion_matrix'] = confusion_matrix(y, pred, labels=range(len(CLASSES))).tolist()
    result['per_class'] = classification_report(y, pred, labels=range(len(CLASSES)), target_names=CLASSES, output_dict=True, zero_division=0)
    rows = []
    for i, row in enumerate(frame.itertuples()):
        im = dataset.images[i]
        gray = np.asarray(im.convert('L'), dtype=float)
        brightness = float(gray.mean())
        edge = float(np.asarray(im.convert('L').filter(ImageFilter.FIND_EDGES))[2:-2, 2:-2].mean())
        correct = y[i] == pred[i]
        analysis = 'Correct' if correct else 'Misclassified; possible visual overlap between stages; human review required'
        if not correct and brightness < 45:
            analysis = 'Misclassified; low brightness measured; contribution to error unverified'
        elif not correct and edge < 5:
            analysis = 'Misclassified; low edge detail measured; blur or uniform background possible'
        rows.append(dict(path=row.path, true_label=row.label, prediction=CLASSES[pred[i]],
            confidence=float(max(probs[i])), correct=correct, analysis=analysis,
            brightness=brightness, edge_detail=edge))
    pd.DataFrame(rows).to_csv(folder / 'test_predictions.csv', index=False)
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = np.array(result['confusion_matrix'])
    ax.imshow(cm, cmap='Greens')
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set(xticks=range(len(CLASSES)), yticks=range(len(CLASSES)), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel='Predicted label', ylabel='True label', title='Held-out test confusion matrix')
    fig.tight_layout()
    fig.savefig(folder / 'confusion_matrix.png', dpi=140)
    plt.close(fig)
    return result

def main():
    global CLASSES, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--fruit', choices=list(FRUITS), default='banana')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    CLASSES = FRUITS[args.fruit]
    data_dir, OUT = paths(args.fruit)
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(data_dir / 'manifest.csv')
    digest = hashlib.sha256((data_dir / 'manifest.csv').read_bytes()).hexdigest()
    frames = {s: frame[frame.split == s].reset_index(drop=True) for s in ['train', 'val', 'test']}
    datasets = {s: Images(f) for s, f in frames.items()}
    
    seed_all(42)
    augmented = Images(frames['train'], augment=True)
    pairs = [augmented[i] for i in range(len(augmented))]
    train_aug = TensorDataset(torch.stack([x for x, _ in pairs]), torch.tensor([y for _, y in pairs]))
    results = []
    for config in EXPERIMENTS:
        seed_all(42)
        folder = OUT / config['id']
        folder.mkdir(exist_ok=True)
        config = dict(config, fruit=args.fruit, epochs=args.epochs, batch_size=args.batch_size, seed=42,
            optimizer='Adam', weight_decay=0., augmentation='Horizontal flip + rotation ±12 degrees',
            image_size=SIZE, manifest_sha256=digest, device='cpu', torch_version=torch.__version__)
        print('\nEXPERIMENT', json.dumps(config), flush=True)
        model = build(config['architecture'], pretrained=config['transfer'], num_classes=len(CLASSES))
        start = time.perf_counter()
        if config['transfer']:
            head, features = head_and_features(model, config['architecture'])
            for p in features.parameters():
                p.requires_grad_(False)
            features.eval()
            cached = {}
            for split in ['train', 'val']:
                xx, yy = [], []
                
                source = train_aug if split == 'train' else datasets[split]
                with torch.inference_mode():
                    for x, y in DataLoader(source, batch_size=args.batch_size):
                        xx.append(features(x).clone())
                        yy.append(y)
                cached[split] = TensorDataset(torch.cat(xx), torch.cat(yy))
                print('Cached features', split, len(source), flush=True)
            trainable = head
            loaders = {s: DataLoader(cached[s], batch_size=args.batch_size, shuffle=s == 'train') for s in cached}
            config['augmentation_sampling'] = 'One fixed seeded augmented view per train image, identical across all experiments.'
        else:
            trainable = model
            loaders = dict(train=DataLoader(train_aug, batch_size=args.batch_size, shuffle=True),
                           val=DataLoader(datasets['val'], batch_size=args.batch_size))
            config['augmentation_sampling'] = 'One fixed seeded augmented view per train image, identical across all experiments.'
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(trainable.parameters(), lr=config['lr'])
        history, best_f1, best_epoch = [], -1., 0
        for epoch in range(1, args.epochs + 1):
            tr = pass_epoch(trainable, loaders['train'], criterion, optimizer)
            va = pass_epoch(trainable, loaders['val'], criterion)
            entry = dict(epoch=epoch, **{'train_' + k: tr[k] for k in ['loss', 'accuracy', 'f1']},
                         **{'val_' + k: va[k] for k in ['loss', 'accuracy', 'f1']},
                         gradient_mean=tr['gradient_mean'], gradient_max=tr['gradient_max'])
            history.append(entry)
            pd.DataFrame(history).to_csv(folder / 'history.csv', index=False)
            print(json.dumps(entry), flush=True)
            if va['f1'] > best_f1:
                best_f1, best_epoch = va['f1'], epoch
                torch.save(dict(state_dict=model.state_dict(), config=config, classes=CLASSES), folder / 'best.pt')
        training_seconds = time.perf_counter() - start
        checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        result = dict(config=config, best_epoch=best_epoch, validation_f1=best_f1,
            training_seconds=training_seconds, parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            diagnostics=diagnostics(history), history=history,
            test=evaluate(model, datasets['test'], frames['test'], folder, args.batch_size))
        curves(history, folder)
        (folder / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        results.append(result)
        print('TEST', config['id'], result['test'], flush=True)
    
    selected = max(results, key=lambda r: r['validation_f1'])['config']['id']
    baseline = results[0]['test']
    for r in results:
        r['effect_vs_baseline'] = {k: r['test'][k] - baseline[k] for k in ['accuracy', 'precision', 'recall', 'f1']}
    summary = dict(selected_model=selected, selection_rule='Highest validation macro F1; test never used for selection',
                   fruit=args.fruit, classes=CLASSES, experiments=results, audit=json.loads((data_dir / 'audit.json').read_text()))
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    pd.DataFrame([dict(experiment=r['config']['id'], **{k:r['test'][k] for k in ['accuracy','precision','recall','f1']},
        validation_f1=r['validation_f1'], best_epoch=r['best_epoch'], training_seconds=r['training_seconds']) for r in results]).to_csv(OUT / 'comparison.csv', index=False)
    lines = ['# Fruit maturity experiment report', '', f'Validation-selected model: **{selected}**.', '',
        'All metrics below are measured on the same frozen test manifest. Single seed; differences are descriptive, not significance claims.', '',
        '| Experiment | Accuracy | Macro precision | Macro recall | Macro F1 | Δ F1 vs baseline |',
        '|---|---:|---:|---:|---:|---:|']
    for r in results:
        t = r['test']
        lines.append(f"| {r['config']['name']} | {t['accuracy']:.4f} | {t['precision']:.4f} | {t['recall']:.4f} | {t['f1']:.4f} | {r['effect_vs_baseline']['f1']:+.4f} |")
    for r in results:
        lines += ['', '## ' + r['config']['name'], '', r['config']['changed'], '', r['config']['why'], '',
                  f"Best epoch: {r['best_epoch']}; validation macro F1: {r['validation_f1']:.4f}.", '',
                  f"![Training curves]({r['config']['id']}/curves.png)", '',
                  f"![Confusion matrix]({r['config']['id']}/confusion_matrix.png)"]
        lines += ['', *[f"- {d['topic']}: {'indicator present' if d['flag'] else 'indicator absent'}. {d['detail']}" for d in r['diagnostics']]]
    lines += ['', '## Interpretation and limitations', '',
        'The lower-LR condition isolates learning rate. Transfer conditions change architecture, initialization and trainable layers together. Do not attribute those differences solely to architecture. Every experiment uses the same fixed seeded augmented training images; validation/test images are unaugmented.', '',
        'Rotten is a dataset quality/spoilage category rather than a precise biological maturity stage. This model assesses appearance, not food safety.', '',
        'The public dataset includes related images and no physical-fruit/session identifiers. Pixel/source/dHash deduplication reduces leakage but cannot establish fruit-independent generalization. Repeated seeds and an independently collected grouped test set are needed for research conclusions.', '',
        'Inspect test_predictions.csv for every image: true label, prediction, confidence and error-analysis hypothesis. Brightness/edge measures are indicators, not confirmed causes. Softmax scores are uncalibrated. Images of a different fruit from the selected model are outside the evaluated task.', '',
        f'Fruit: {args.fruit}. Data audit: {data_dir.relative_to(ROOT).as_posix()}/audit.json. Split strategy and limitations are recorded there; class prevalence is dataset-specific.']
    (OUT / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('DONE. Selected:', selected, flush=True)

if __name__ == '__main__':
    main()
