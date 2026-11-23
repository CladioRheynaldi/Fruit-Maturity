import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from fruits import ROOT, FRUITS, paths

SOURCE = {
    'apple': ('apple_orange', 'asadullahprl/fruits-ripeness-classification-dataset'),
    'orange': ('apple_orange', 'asadullahprl/fruits-ripeness-classification-dataset'),
    'strawberry': ('strawberry', 'hamza202404/strawberry-ripeness-classification-dataset'),
    'pineapple': ('pineapple', 'adhilpk/pineapple'),
}

def entries(fruit):
    folder = ROOT / 'data/sources' / SOURCE[fruit][0] / 'raw'
    if fruit in ('apple', 'orange'):
        for p in sorted(folder.rglob('*')):
            if p.is_file() and p.name.lower().startswith(fruit + '_'):
                label = p.parent.name.lower().replace('overipe', 'overripe')
                if label in FRUITS[fruit]:
                    yield p, label, 'test' if 'Test' in p.parts else 'train', p.stem
    elif fruit == 'strawberry':
        mapping = {'Unripe': 'unripe', 'Semiripe': 'semi-ripe', 'Fullripe': 'ripe'}
        for row in pd.read_csv(folder / 'stra.csv').itertuples():
            if row.ripeness_type not in mapping:
                raise ValueError('Unknown strawberry label: ' + str(row.ripeness_type))
            p = folder / 'images/images' / row.image_name
            yield p, mapping[row.ripeness_type], 'unsplit', p.name.split('.rf.')[0]
    else:
        crops = paths(fruit)[0] / 'crops'
        crops.mkdir(parents=True, exist_ok=True)
        for mapping in folder.rglob('classes.txt'):
            if mapping.read_text().splitlines() != ['UNMATURED_PINEAPPLE', 'MATURED_PINEAPPLE']:
                raise ValueError('Unexpected pineapple class order')
        for p in sorted(folder.rglob('*')):
            if p.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            annotation = p.with_suffix('.txt')
            if not annotation.exists():
                continue
            boxes = []
            for line in annotation.read_text().splitlines():
                values = line.split()
                if len(values) != 5:
                    raise ValueError('Invalid YOLO annotation: ' + str(annotation))
                cls, x, y, w, h = map(float, values)
                if cls not in (0, 1) or not all(0 <= v <= 1 for v in (x,y,w,h)) or min(w,h) <= 0:
                    raise ValueError('Invalid bounding box: ' + str(annotation))
                boxes.append((int(cls), x,y,w,h))
            if not boxes:
                continue
            
            cls, x,y,w,h = max(boxes, key=lambda box: box[3]*box[4])
            with Image.open(p) as im:
                im = im.convert('RGB')
                width,height = im.size
                bounds = (max(0,int((x-w*.55)*width)),max(0,int((y-h*.55)*height)),
                          min(width,int((x+w*.55)*width)),min(height,int((y+h*.55)*height)))
                crop = im.crop(bounds)
                target = crops / (hashlib.sha256(p.relative_to(folder).as_posix().encode()).hexdigest()[:20]+'.png')
                crop.save(target)
            source = re.sub(r'_edited.*$', '', p.stem)
            yield target, ['unripe','ripe'][cls], 'test' if 'test' in p.parts else 'train', source

def prepare(fruit):
    folder, _ = paths(fruit)
    folder.mkdir(parents=True, exist_ok=True)
    rows, rejected = [], []
    for p, label, split, source in entries(fruit):
        try:
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                im.load()
                if min(im.size) < 32:
                    raise ValueError('Too small: minimum side <32')
                sha = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
                gray = np.asarray(im.resize((9, 8)).convert('L'))
                dhash = np.packbits(gray[:, 1:] > gray[:, :-1]).tobytes().hex()
            rows.append(dict(path=p.relative_to(ROOT).as_posix(), label=label, split=split,
                             source=source, sha256=sha, dhash=dhash))
        except (OSError, ValueError) as error:
            rejected.append(dict(path=str(p), reason=str(error)))
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError('No valid images for ' + fruit)
    original = df.groupby(['split', 'label']).size().unstack(fill_value=0).to_dict('index')
    parents = list(range(len(df)))
    def find(i):
        while i != parents[i]:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for key in ['source', 'sha256', 'dhash']:
        seen = {}
        for i, value in enumerate(df[key]):
            if value in seen:
                parents[find(i)] = find(seen[value])
            else:
                seen[value] = i
    df['group'] = [find(i) for i in range(len(df))]
    clean = []
    for _, group in df.groupby('group'):
        if group.label.nunique() > 1:
            rejected.extend(dict(path=r.path, reason='Duplicate group label conflict') for r in group.itertuples())
            continue
        group = group.assign(priority=group.split.map({'test':0, 'train':1, 'unsplit':1})).sort_values('priority')
        clean.append(group.iloc[0].drop('priority'))
        rejected.extend(dict(path=r.path, reason='Duplicate/source/dHash match') for r in group.iloc[1:].itertuples())
    df = pd.DataFrame(clean).reset_index(drop=True)
    
    if fruit in ('apple', 'orange', 'pineapple'):
        train = df[df.split == 'train']
        train_idx, val_idx = train_test_split(train.index, test_size=.20, stratify=train.label, random_state=42)
        df.loc[val_idx, 'split'] = 'val'
    else:
        train_idx, hold = train_test_split(df.index, test_size=.30, stratify=df.label, random_state=42)
        val_idx, test_idx = train_test_split(hold, test_size=.5, stratify=df.loc[hold, 'label'], random_state=42)
        df.loc[train_idx, 'split'] = 'train'
        df.loc[val_idx, 'split'] = 'val'
        df.loc[test_idx, 'split'] = 'test'
    df.to_csv(folder / 'clean_manifest.csv', index=False)
    used = []
    for (split, label), group in df.groupby(['split','label']):
        used.append(group.sample(n=min(len(group), 300) if split == 'train' else len(group), random_state=42))
    manifest = pd.concat(used).sort_values(['split','label','path'])
    counts = manifest.groupby(['split','label']).size().unstack(fill_value=0).reindex(columns=FRUITS[fruit], fill_value=0)
    if counts.shape[0] != 3 or (counts < 3).any().any():
        raise ValueError('Insufficient class/split support: ' + str(counts))
    manifest.to_csv(folder / 'manifest.csv', index=False)
    pd.DataFrame(rejected, columns=['path','reason']).to_csv(folder / 'rejected.csv', index=False)
    metadata = json.loads((ROOT / 'data/sources' / SOURCE[fruit][0] / 'metadata.json').read_text(encoding='utf-8'))
    audit = dict(fruit=fruit, source='https://www.kaggle.com/datasets/' + SOURCE[fruit][1],
                 license=metadata.get('licenseName', 'See source metadata'), downloaded_images=len(rows),
                 retained_after_cleaning=len(df), used=len(manifest), rejected=len(rejected),
                 original_counts=original, counts=counts.to_dict('index'), seed=42,
                 split_policy='Preserve official test; stratified 20% of cleaned training for validation.' if fruit in ('apple','orange','pineapple') else 'Seeded stratified 70/15/15 split after duplicate grouping and removal.',
                 preprocessing='EXIF orientation, RGB, 128x128 bicubic resize, [0,1] scaling then ImageNet normalization; same seeded flip/rotation training views for every experiment. Tokenization not applicable.',
                 missing_data='Exclude and log invalid/missing images, reject unknown labels; do not impute maturity labels.',
                 limitation='Single-seed public-data prototype. Exact/source/dHash grouping cannot remove all related frames or shared physical fruits; no complete fruit/session IDs. Image-level evaluation may overestimate external generalization. Training caps are up to 300 per class, not necessarily balanced.')
    if fruit == 'pineapple':
        audit['label_mapping'] = {'UNMATURED_PINEAPPLE': 'unripe', 'MATURED_PINEAPPLE': 'ripe'}
        audit['preprocessing'] = 'Largest YOLO annotated pineapple crop with 5% border. Source labels indicate maturity, not measured internal sweetness. ' + audit['preprocessing']
        audit['limitation'] += ' Pineapple inputs must be closely framed single-fruit photos; this classifier does not detect/crop fruit at inference. Related sequential field photos may remain across splits.'
    (folder / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(fruit, counts.to_dict('index'), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fruits', nargs='*', default=['apple','orange','strawberry'], choices=list(SOURCE))
    args = parser.parse_args()
    for fruit in args.fruits:
        prepare(fruit)
