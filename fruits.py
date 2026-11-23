from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRUITS = {
    'banana': ['unripe', 'ripe', 'overripe', 'rotten'],
    'apple': ['unripe', 'ripe', 'overripe'],
    'orange': ['unripe', 'ripe', 'overripe'],
    'strawberry': ['unripe', 'semi-ripe', 'ripe'],
    'pineapple': ['unripe', 'ripe'],
}

def paths(fruit):
    if fruit not in FRUITS:
        raise ValueError('Unsupported fruit: ' + fruit)
    suffix = Path() if fruit == 'banana' else Path('fruits') / fruit
    return ROOT / 'data' / suffix, ROOT / 'artifacts' / suffix
