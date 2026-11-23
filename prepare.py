import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from models import CLASSES

ROOT = Path(__file__).resolve().parent
URL = 'https://huggingface.co/datasets/Sapek007/banana-ripeness/resolve/main/banana_ripeness_dataset.zip'
SHA256 = '2b12641883bbf13e782727e3deb9a3bf141a26fecb832bc045e508e319014567'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--train-per-class', type=int, default=400)
    parser.add_argument('--eval-per-class', type=int, default=150)
    args = parser.parse_args()
    data = ROOT / 'data'
    data.mkdir(exist_ok=True)
    if args.download:
        archive = data / 'dataset.zip'
        if not archive.exists():
            urllib.request.urlretrieve(URL, archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
            raise ValueError('Dataset checksum mismatch')
        target = (data / 'raw').resolve()
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                if not (target / item.filename).resolve().is_relative_to(target):
                    raise ValueError('Unsafe ZIP path')
            z.extractall(target)
    records, rejected = [], []
    for split in ['test', 'valid', 'train']:
        for path in sorted((data / 'raw' / split).rglob('*')):
            if not path.is_file():
                continue
            if path.parent.name not in CLASSES:
                rejected.append({'path': str(path), 'reason': 'Missing or unknown label / non-image metadata'})
                continue
            try:
                with Image.open(path) as im:
                    im = ImageOps.exif_transpose(im).convert('RGB')
                    im.load()
                    if min(im.size) < 32:
                        raise ValueError('Image smaller than 32 pixels')
                    digest = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
                    gray = np.asarray(im.resize((9, 8)).convert('L'))
                    dhash = np.packbits(gray[:, 1:] > gray[:, :-1]).tobytes().hex()
                    records.append(dict(path=path.relative_to(ROOT).as_posix(), label=path.parent.name,
                        split='val' if split == 'valid' else split, sha256=digest, dhash=dhash,
                        source=path.name.split('.rf.')[0]))
            except Exception as e:
                rejected.append({'path': str(path), 'reason': str(e)})
    if not records:
        raise ValueError('No labeled images found. Run prepare.py --download first.')
    df = pd.DataFrame(records)
    original = df.groupby(['split', 'label']).size().unstack(fill_value=0).to_dict('index')
    
    parent = list(range(len(df)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for key in ['sha256', 'source', 'dhash']:
        seen = {}
        for i, value in enumerate(df[key]):
            if value in seen:
                parent[find(i)] = find(seen[value])
            else:
                seen[value] = i
    df['group'] = [find(i) for i in range(len(df))]
    keep = []
    for _, group in df.groupby('group', sort=False):
        if group.label.nunique() > 1:
            for row in group.itertuples():
                rejected.append({'path': row.path, 'reason': 'Conflicting labels in duplicate group'})
            continue
        
        keep.append(group.index[0])
        for row in group.iloc[1:].itertuples():
            rejected.append({'path': row.path, 'reason': 'Duplicate/source variant/perceptual-hash match'})
    clean = df.loc[keep].copy()
    clean.to_csv(data / 'clean_manifest.csv', index=False)
    selected = []
    for (split, label), group in clean.groupby(['split', 'label']):
        cap = args.train_per_class if split == 'train' else args.eval_per_class
        selected.append(group.sample(n=min(len(group), cap) if cap > 0 else len(group), random_state=42))
    manifest = pd.concat(selected).sort_values(['split', 'label', 'path'])
    counts = manifest.groupby(['split', 'label']).size().unstack(fill_value=0)
    if counts.shape != (3, 4) or (counts < 5).any().any():
        raise ValueError(f'Insufficient class support after cleaning: {counts}')
    manifest.to_csv(data / 'manifest.csv', index=False)
    pd.DataFrame(rejected).to_csv(data / 'rejected.csv', index=False)
    audit = dict(source='https://huggingface.co/datasets/Sapek007/banana-ripeness',
        license='CC BY 4.0 as declared by dataset publisher; original Roboflow attribution in data/raw/README.roboflow.txt',
        downloaded_images=len(df), retained_after_cleaning=len(clean), rejected=len(rejected),
        used=len(manifest), original_counts=original, counts=counts.to_dict('index'),
        seed=42, train_cap_per_class=args.train_per_class, eval_cap_per_class=args.eval_per_class,
        split_policy='Preserve upstream split; one representative per connected duplicate group; test priority, then validation. Seeded class caps form a computational subset.',
        missing_data='Unreadable images and unknown/missing labels are excluded and logged; pixels/labels are never imputed.',
        limitation='Identical dHash is a conservative duplicate heuristic, not proof of identity. Near-duplicate frames with different hashes and shared physical fruits can remain. No fruit/session IDs supplied; external grouped evaluation is required.',
        preprocessing='RGB + EXIF orientation; 128x128 resize; ImageNet mean/std normalization. Training-only flip/rotation when enabled. Tokenization is not applicable to CNN images.')
    (data / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))

if __name__ == '__main__':
    main()
