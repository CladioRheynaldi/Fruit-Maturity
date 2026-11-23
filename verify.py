import json
import unittest
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from fruits import FRUITS, paths
from app import app

ROOT = Path(__file__).resolve().parent

class ProjectChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_all_fruits_data_and_leakage(self):
        for fruit, classes in FRUITS.items():
            data_dir, artifacts = paths(fruit)
            self.assertTrue(data_dir.exists(), f"Missing data directory for {fruit}")
            self.assertTrue(artifacts.exists(), f"Missing artifacts directory for {fruit}")
            manifest = pd.read_csv(data_dir / 'manifest.csv')
            for key in ['sha256', 'source', 'dhash', 'group']:
                if key in manifest.columns:
                    self.assertEqual(int(manifest.groupby(key).split.nunique().max()), 1,
                                     f"Data leakage detected in {fruit} along {key}")
            for _, frame in manifest.groupby('split'):
                self.assertEqual(set(frame.label), set(classes),
                                 f"Class mismatch in {fruit} split")

    def test_metrics_match_saved_predictions_for_all_fruits(self):
        for fruit in FRUITS:
            data_dir, artifacts = paths(fruit)
            summary = json.loads((artifacts / 'summary.json').read_text(encoding='utf-8'))
            manifest = pd.read_csv(data_dir / 'manifest.csv')
            test_manifest = manifest[manifest.split == 'test']
            for run in summary['experiments']:
                ident = run['config']['id']
                rows = pd.read_csv(artifacts / ident / 'test_predictions.csv')
                self.assertEqual(rows.path.tolist(), test_manifest.path.tolist(),
                                 f"Path order mismatch in {fruit} {ident}")
                self.assertAlmostEqual(accuracy_score(rows.true_label, rows.prediction), run['test']['accuracy'],
                                       places=4, msg=f"Accuracy mismatch in {fruit} {ident}")
                self.assertAlmostEqual(f1_score(rows.true_label, rows.prediction, average='macro'), run['test']['f1'],
                                       places=4, msg=f"Macro F1 mismatch in {fruit} {ident}")
            expected = max(summary['experiments'], key=lambda r: r['validation_f1'])['config']['id']
            self.assertEqual(expected, summary['selected_model'])

    def test_web_and_checkpoint_inference_all_fruits(self):
        res = self.client.get('/')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Ripeness Lab', res.data)
        res.close()

        for fruit in FRUITS:
            data_dir, artifacts = paths(fruit)
            summary = json.loads((artifacts / 'summary.json').read_text(encoding='utf-8'))
            manifest = pd.read_csv(data_dir / 'manifest.csv')
            test_manifest = manifest[manifest.split == 'test']
            test_sample_bytes = (ROOT / test_manifest.iloc[0].path).read_bytes()

            
            res = self.client.post(f'/api/predict?fruit={fruit}', data=test_sample_bytes, content_type='image/jpeg')
            self.assertEqual(res.status_code, 200, f"Inference failed on {fruit}")
            data = res.get_json()
            self.assertEqual(data['fruit'], fruit)
            self.assertIn(data['label'], FRUITS[fruit])
            self.assertAlmostEqual(sum(data['probabilities'].values()), 1.0, places=4)

    def test_invalid_input(self):
        for suffix, payload in [('mobilenet', b'not an image'), ('missing', b'test')]:
            res = self.client.post('/api/predict?model=' + suffix, data=payload)
            self.assertEqual(res.status_code, 400)
            self.assertIn('error', res.get_json())

    def test_upload_limits(self):
        for payload in [b'', b'x' * (10 * 1024 * 1024 + 1)]:
            res = self.client.post('/api/predict', data=payload)
            self.assertEqual(res.status_code, 413)
            self.assertIn('error', res.get_json())

    def test_routes_and_file_boundaries(self):
        for path in ['/style.css', '/app.js', '/sample/0?fruit=apple', '/sample/0?fruit=banana',
                     '/artifacts/comparison.csv', '/artifacts/mobilenet/curves.png',
                     '/artifacts/fruits/apple/comparison.csv', '/artifacts/fruits/pineapple/mobilenet/curves.png',
                     '/api/fruits', '/api/summary?fruit=banana', '/api/summary?fruit=strawberry',
                     '/api/errors?fruit=orange&model=mobilenet']:
            res = self.client.get(path)
            self.assertEqual(res.status_code, 200, path)
            self.assertEqual(res.headers['Cache-Control'], 'no-store')
            res.close()
        for path in ['/artifacts/mobilenet/best.pt', '/artifacts/../app.py',
                     '/%2e%2e/app.py', '/sample/999999', '/sample/-1', '/not-found']:
            res = self.client.get(path)
            self.assertEqual(res.status_code, 404, path)
            self.assertIn('error', res.get_json())
        self.assertEqual(self.client.get('/api/errors?model=missing').status_code, 400)
        self.assertEqual(self.client.get('/api/predict').status_code, 405)

    def test_api_fruits_endpoint(self):
        res = self.client.get('/api/fruits')
        self.assertEqual(res.status_code, 200)
        items = res.get_json()
        self.assertEqual(len(items), 5)
        self.assertTrue(all(item['ready'] for item in items))

if __name__ == '__main__':
    unittest.main(verbosity=2)
