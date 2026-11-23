import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
SOURCES = {
    'apple_orange': 'asadullahprl/fruits-ripeness-classification-dataset',
    'strawberry': 'hamza202404/strawberry-ripeness-classification-dataset',
    'pineapple': 'adhilpk/pineapple',
    'strawberry_boxes': 'mahyeks/multi-class-strawberry-ripeness-detection-dataset',
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sources', nargs='*', default=list(SOURCES))
    args = parser.parse_args()
    for name in args.sources:
        ref = SOURCES[name]
        folder = ROOT / 'data' / 'sources' / name
        folder.mkdir(parents=True, exist_ok=True)
        metadata = json.load(urllib.request.urlopen('https://www.kaggle.com/api/v1/datasets/view/' + ref, timeout=60))
        (folder / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        archive = folder / 'download.zip'
        if not archive.exists():
            temporary = folder / 'download.part'
            urllib.request.urlretrieve('https://www.kaggle.com/api/v1/datasets/download/' + ref, temporary)
            temporary.replace(archive)
        with zipfile.ZipFile(archive) as z:
            destination = (folder / 'raw').resolve()
            for item in z.infolist():
                if not (destination / item.filename).resolve().is_relative_to(destination):
                    raise ValueError('Unsafe archive path')
            z.extractall(destination)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        (folder / 'sha256.txt').write_text(digest, encoding='utf-8')
        print(name, 'downloaded', archive.stat().st_size, 'bytes', digest, flush=True)

if __name__ == '__main__':
    main()
