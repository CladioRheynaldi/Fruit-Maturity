import argparse
import io
import json
import threading
from pathlib import Path
from flask import Flask, jsonify, request, send_file, abort
from werkzeug.exceptions import HTTPException
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
import torch
from models import build, transform, CLASSES
from fruits import FRUITS, paths

ROOT = Path(__file__).resolve().parent
LOCK = threading.Lock()
CACHE = {}
IDS = {'baseline', 'cnn_low_lr', 'resnet18', 'mobilenet'}
torch.set_num_threads(4)

def summary(fruit='banana'):
    data_dir, artifacts = paths(fruit)
    p = artifacts / 'summary.json'
    if p.exists():
        result = json.loads(p.read_text(encoding='utf-8'))
        result['fruit'] = fruit
        result['artifact_base'] = '/' + artifacts.relative_to(ROOT).as_posix()
        return result
    experiments = []
    for name in sorted(IDS):
        folder = artifacts / name
        if (folder / 'results.json').exists():
            experiments.append(json.loads((folder / 'results.json').read_text(encoding='utf-8')))
    return dict(training=True, fruit=fruit, experiments=experiments, classes=FRUITS[fruit],
                artifact_base='/' + artifacts.relative_to(ROOT).as_posix(),
                audit=json.loads((data_dir / 'audit.json').read_text(encoding='utf-8')) if (data_dir / 'audit.json').exists() else {})

def predict(payload, model_id, fruit='banana'):
    _, artifacts = paths(fruit)
    if model_id not in IDS:
        raise ValueError('Unknown model')
    checkpoint = artifacts / model_id / 'best.pt'
    if not checkpoint.exists():
        raise ValueError('This model has not been trained yet')
    with Image.open(io.BytesIO(payload)) as im:
        if im.width * im.height > 20_000_000:
            raise ValueError('Please use an image below 20 megapixels')
        im = ImageOps.exif_transpose(im).convert('RGB')
        tensor = transform()(im).unsqueeze(0)
    with LOCK:
        key = (fruit, model_id)
        stamp = checkpoint.stat().st_mtime_ns
        if key not in CACHE or CACHE[key][0] != stamp:
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            classes = state['classes']
            if classes != FRUITS[fruit] or state['config'].get('fruit', 'banana') != fruit:
                raise ValueError('Checkpoint fruit/classes do not match the selected fruit')
            model = build(state['config']['architecture'], num_classes=len(classes))
            model.load_state_dict(state['state_dict'])
            model.eval()
            CACHE[key] = (stamp, model, classes)
        classes = CACHE[key][2]
        with torch.inference_mode():
            probs = CACHE[key][1](tensor).softmax(1)[0].tolist()
    index = max(range(len(probs)), key=probs.__getitem__)
    return dict(fruit=fruit, label=classes[index], confidence=probs[index], model=model_id,
                probabilities=dict(zip(classes, probs)), uncertain=probs[index] < .65,
                note=f'Scores are uncalibrated. Selected fruit: {fruit}. Fruit identity is user-supplied; unrelated images can still receive confident predictions.')

app = Flask(__name__, static_folder=None)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
app.json.sort_keys = False


@app.after_request
def response_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Cache-Control'] = 'no-store'
    return response


@app.errorhandler(HTTPException)
def http_error(error):
    response = error.get_response()
    message = ('Image must be between 1 byte and 10 MB'
               if error.code == 413 else error.description)
    response.data = app.json.dumps({'error': message})
    response.content_type = 'application/json'
    return response


def public_file(directory, filename, suffixes=None):
    directory = directory.resolve()
    path = (directory / filename).resolve()
    if (not path.is_relative_to(directory) or not path.is_file()
            or (suffixes is not None and path.suffix not in suffixes)):
        abort(404)
    return send_file(path)


@app.get('/api/summary')
def get_summary():
    return jsonify(summary(selected_fruit()))


def selected_fruit():
    fruit = request.args.get('fruit', 'banana')
    if fruit not in FRUITS:
        abort(400, description='Unsupported fruit')
    return fruit


@app.get('/api/fruits')
def get_fruits():
    return jsonify([dict(id=fruit, name=fruit.title(), classes=classes,
                         ready=(paths(fruit)[1] / 'summary.json').exists())
                    for fruit, classes in FRUITS.items()])


@app.get('/api/errors')
def get_errors():
    _, artifacts = paths(selected_fruit())
    model = request.args.get('model', 'baseline')
    if model not in IDS:
        return jsonify(error='Unknown model'), 400
    path = artifacts / model / 'test_predictions.csv'
    return jsonify(pd.read_csv(path).to_dict('records') if path.exists() else [])


@app.get('/sample/<int:index>')
def get_sample(index):
    data_dir, _ = paths(selected_fruit())
    frame = pd.read_csv(data_dir / 'manifest.csv')
    frame = frame[frame.split == 'test'].reset_index(drop=True)
    if index >= len(frame):
        abort(404, description='Sample not found')
    return public_file(ROOT / 'data', str((ROOT / frame.iloc[index].path).relative_to(ROOT / 'data')))


@app.get('/artifacts/<path:filename>')
def get_artifact(filename):
    return public_file(ROOT / 'artifacts', filename, {'.png', '.csv', '.json', '.md'})


@app.get('/')
def index():
    return public_file(ROOT / 'web', 'index.html')


@app.get('/<filename>')
def get_static(filename):
    return public_file(ROOT / 'web', filename)


@app.post('/api/predict')
def post_predict():
    fruit = selected_fruit()
    payload = request.get_data(cache=False)
    if not payload:
        abort(413)
    model = request.args.get('model')
    if model is None:
        model = summary(fruit).get('selected_model', 'baseline')
    try:
        return jsonify(predict(payload, model, fruit))
    except (ValueError, UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        return jsonify(error='Cannot predict: ' + str(error)), 400

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    print(f'Fruit maturity app: http://127.0.0.1:{args.port}', flush=True)
    app.run(host='127.0.0.1', port=args.port, debug=False, use_reloader=False)
