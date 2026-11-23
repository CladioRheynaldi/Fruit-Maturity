# Ripeness Lab — Deep Learning for Fruit Maturity Classification

## Clone this repository

```powershell
git clone https://github.com/CladioRheynaldi/Fruit-Maturity.git
cd Fruit-Maturity
python -m pip install -r requirements.txt
```

This repository contains source code and setup documentation. Datasets, trained checkpoints, evaluation outputs, PDFs, caches, and logs are excluded from Git. Download the datasets and train the models using the commands below before running inference or `verify.py` on a fresh clone. The report scripts additionally require `python -m pip install -r requirements-report.txt` and their saved report evidence files. Local Windows paths shown in the documentation refer to the original project location; use your clone directory instead.

A working PyTorch fruit maturity classification project with four completed experiments per fruit and a local web application. The application supports five fruits: **Banana, Apple, Orange, Strawberry, and Pineapple**.

**Implementation status:** Full multi-fruit classification, data preparation, duplicate cleaning, training, evaluation, Flask web application, and technical documentation are implemented and verified across all five fruits. A comprehensive PDF technical report is generated at `Fruit_Maturity_Technical_Report.pdf`.

## Project objective

Predict a fruit's visible maturity category from an RGB photograph and compare deep learning approaches using a reproducible experiment. The application accepts an image, preprocesses it, performs inference with a trained checkpoint, and displays the predicted class and class scores.

This is an image classification task: it predicts one label for the whole image. The current implementation does not locate individual fruits with bounding boxes or segmentation masks.

## Features

- Image upload and an included held-out sample for testing the application.
- Four selectable trained experiments across three architectures: a custom CNN, ResNet18 and MobileNetV3 Small.
- Dataset cleaning, duplicate handling, fixed train/validation/test manifests and training augmentation.
- Accuracy, macro precision, macro recall, macro F1 and per-class evaluation.
- Training/validation curves, confusion matrices and diagnostic indicators.
- Test-image thumbnails, true labels, predictions and error-analysis hypotheses.
- Downloadable metrics and prediction CSVs.

## Technology

| Component | Implementation |
|---|---|
| Deep learning | PyTorch and torchvision |
| Image processing | Pillow, torchvision transforms |
| Evaluation and data handling | scikit-learn, pandas, NumPy |
| Charts | Matplotlib |
| Web framework | Flask 3.1+; routes, request validation, JSON responses and file serving |
| Local development server | Flask's Werkzeug development server, started by `app.run()` |
| Web interface | HTML, CSS and JavaScript |
| Execution | CPU training and inference |

The recorded training environment used Python 3.13.5, PyTorch 2.10.0+cpu and torchvision 0.25.0+cpu. Dependency ranges are listed in [requirements.txt](requirements.txt); they are not an exact environment lockfile.

## System architecture and framework requirement

The web application uses **Flask**, one of the permitted frameworks (Flask, Django, Streamlit or FastAPI). The previous implementation used Python's `ThreadingHTTPServer` directly, which did not meet that framework requirement. The backend has been migrated to a Flask application in [app.py](app.py).

```text
Browser: HTML / CSS / JavaScript
    | GET page, assets, reports and test samples
    | POST /api/predict with image bytes
    v
Flask application (app.py)
    | Routes, upload limits, JSON errors and responses
    | Image preprocessing with Pillow / torchvision
    v
PyTorch inference (models.py + saved best.pt checkpoint)
    | Four class scores and predicted maturity label
    v
Flask JSON response -> Browser result display

Offline workflow:
prepare.py -> cleaned manifests -> train.py -> checkpoints + evaluation artifacts
                                                    |
                                                    v
                                           Flask reads saved outputs
```

Flask is the web framework; PyTorch remains the deep learning framework. HTML/CSS/JavaScript provide the browser interface. These components serve different roles and work together within the stated web-framework restriction.

Flask uses Werkzeug for its local development server. That is part of Flask's normal runtime setup, not an additional application framework selected in place of Flask. The remaining `threading.Lock` protects shared model inference and cache access; it is a synchronization utility, not a web framework. There is no direct `ThreadingHTTPServer` or `BaseHTTPRequestHandler` implementation in the updated backend.

The existing API URLs, trained checkpoints and preprocessing are preserved, so this migration does not require retraining. The recorded model scores below are the original experiment results. `python app.py` starts Flask on localhost with debug mode and automatic reload disabled; this is the local coursework/demo setup.

References: [Flask application and routing](https://flask.palletsprojects.com/en/stable/quickstart/) and [Flask test client](https://flask.palletsprojects.com/en/stable/testing/).

## Start the web app

```powershell
cd "C:\Users\Cladio\Documents\ASLAB\Deep Learning\Projek"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

These commands use the virtual environment directly, so PowerShell script activation is unnecessary. In the commands below, `python` means the interpreter with the project dependencies installed; substitute `.\.venv\Scripts\python.exe` when using this environment.

Open **http://127.0.0.1:8000**. If already running, just open the URL. Upload a photo, choose any trained model, and click **Analyze maturity**, or use **Try a test sample**. The application loads actual saved weights. It processes uploaded bytes in memory without saving the image. No API key is required. The server binds to localhost and is intended for local coursework use.

The four tabs provide image inference, experiment comparisons, training curves and diagnostics, and per-image test analysis with thumbnails. Raw metrics and prediction CSVs can be downloaded.

### Using the application

1. Open the local URL after starting `app.py`.
2. Select a trained inference model. MobileNetV3 is the validation-selected default in the saved run.
3. Upload a JPG, PNG or WebP image, or choose **Try a test sample**. Uploads are limited to 10 MB and 20 megapixels.
4. Click **Analyze maturity** to display the predicted stage and scores for all four classes.
5. Explore **Model comparison**, **Training & diagnostics**, and **Test analysis** for supporting evidence.

Use a clear, close photograph of a banana or a bunch at a similar maturity stage. The interface marks a prediction as uncertain when its highest softmax score is below 0.65. This threshold is a display heuristic, not a calibrated reliability estimate.

Stop a foreground server with `Ctrl+C`. To use a different port, run `python app.py --port 8001` and open `http://127.0.0.1:8001`.

## Model architecture and inference flow

```text
Uploaded image
    -> EXIF orientation and RGB conversion
    -> 128 x 128 bicubic resize
    -> Tensor scaling and ImageNet normalization
    -> Selected trained CNN / ResNet18 / MobileNetV3
    -> Softmax over four maturity labels
    -> Predicted label and class scores in the browser
```

The custom CNN contains three convolution/ReLU/max-pooling blocks with 16, 32 and 64 output channels, followed by adaptive average pooling and a four-class linear layer. ResNet18 uses an ImageNet-pretrained frozen backbone with a newly trained linear head. MobileNetV3 Small uses a frozen pretrained feature backbone with a trainable classifier whose final layer has four outputs.

Training minimizes cross-entropy loss with Adam. All experiments use gradient clipping with a maximum norm of 5. Frozen transfer-model features are cached to reduce CPU training time.

## Completed experiment results

12 epochs each, Adam, batch size 32, seed 42, 128×128 RGB, identical fixed augmented training inputs. All metrics below use the same 518 test images. Precision, recall and F1 use macro averaging.

| Condition | Learning rate | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Baseline: small CNN, random initialization | 0.001 | 85.14% | 85.91% | 85.84% | 0.8553 |
| Experiment 1: same CNN, lower learning rate | 0.0001 | 75.87% | 76.20% | 77.57% | 0.7597 |
| Experiment 2: frozen ImageNet ResNet18 + trained head | 0.001 | 92.08% | 92.75% | 91.98% | 0.9227 |
| Experiment 3: frozen ImageNet MobileNetV3 Small + trained classifier | 0.001 | 95.56% | 95.47% | 95.91% | 0.9563 |

**MobileNetV3 was selected by validation macro F1 (0.9733), not by test scores.** Its chosen checkpoint is epoch 9. The CNN baseline uses epoch 7, lower-LR CNN epoch 12, and ResNet18 epoch 12.

| Experiment | What changed | Why it changed | Measured test macro F1 effect |
|---|---|---|---:|
| Baseline | Small CNN trained from random initialization at LR 0.001 | Establish a reference for learning visual features from scratch | Reference: 0.8553 |
| Lower learning rate | Only LR changes from 0.001 to 0.0001 | Check sensitivity to smaller optimization steps at the same epoch budget | −9.56 percentage points |
| ResNet18 transfer | Architecture, initialization and trainable layers | Evaluate pretrained residual features | +6.74 percentage points |
| MobileNetV3 transfer | Architecture, initialization and trainable layers | Evaluate a compact pretrained feature extractor | +10.10 percentage points |

See [comparison.csv](artifacts/comparison.csv), [summary.json](artifacts/summary.json), and the [full experiment report](artifacts/REPORT.md) for the saved measurements and configurations. These numbers describe the delivered run; rerunning training may produce different results.

- Lowering only the learning rate reduced test F1 by 9.56 percentage points at this fixed budget. This does not show that the smaller learning rate would remain worse with longer training.
- ResNet18 transfer improved F1 by 6.74 points versus baseline.
- MobileNetV3 transfer improved F1 by 10.10 points versus baseline, and 3.36 points versus ResNet18.
- Transfer conditions change architecture, pretrained initialization and which layers are trained together. They are practical model comparisons, not isolated architecture ablations.

## Reproduce training

```powershell
python prepare.py --download
python train.py --epochs 12 --batch-size 32 --threads 8
python verify.py
python app.py
```

Preparation downloads the public archive, verifies its SHA-256, safely extracts it, and writes frozen manifests. Downloading model weights requires internet on the first training run; inference works offline. External web fonts are optional and fall back to system fonts.

For all retained images instead of the documented computational subset:

```powershell
python prepare.py --train-per-class 0 --eval-per-class 0
python train.py --epochs 12
```

This changes the experiment population and overwrites generated results, so archive `artifacts/` and `data/manifest.csv` first if keeping the original run. The current implementation deliberately runs on CPU; frozen-backbone feature caching makes the transfer experiments fast. It records the environment, configuration, random seed and manifest hash with each checkpoint. Reproducibility across hardware/library versions is approximate.

## Data and preparation

Dataset: [Sapek007/banana-ripeness](https://huggingface.co/datasets/Sapek007/banana-ripeness), a mirror of [Roboflow Universe Projects / Banana Ripeness Classification v6](https://universe.roboflow.com/roboflow-universe-projects/banana-ripeness-classification), CC BY 4.0 as declared by both the mirror and bundled dataset metadata. Attribution files are preserved in `data/raw/README.dataset.txt` and `README.roboflow.txt`. The downloaded archive contains 13,478 labeled images, including upstream augmented variants.

1. Decode and verify each image; remove corrupt or too-small images and records with unknown/missing labels. Do not invent missing labels or impute missing images. Log exclusions in `data/rejected.csv`.
2. Form connected duplicate groups from pixel SHA-256, the source filename before `.rf.`, and identical 64-bit difference hashes. Drop groups with conflicting labels; keep one representative from each remaining group, prioritizing test then validation then training. This is conservative and may group some distinct images.
3. Retain 5,261 images after cleaning. Keep upstream split assignments for retained representatives. Select a deterministic computational subset: 400 per training class, at most 150 per validation/test class. The delivered run uses 1,600 train, 600 validation, and 518 test images.
4. Apply EXIF orientation, RGB conversion, 128×128 resize, tensor scaling to [0,1], then ImageNet mean/std normalization.
5. Apply horizontal flips and ±12° rotations to training images only. Generate the same seeded augmented view for every experiment. Transfer features are cached from these views. Validation/test images are unaugmented by our pipeline; upstream transformations are documented by the publisher.
6. Tokenization is not relevant to image CNN models.

Original training counts were 1,902 unripe / 3,522 ripe / 2,349 overripe / 4,020 rotten (max/min ratio 2.11). Class caps balance the training subset. Test support is 105 / 150 / 113 / 150 respectively; macro metrics treat classes equally. Class prevalence in this subset does not represent deployment prevalence.

| Label | Training | Validation | Test |
|---|---:|---:|---:|
| Unripe | 400 | 150 | 105 |
| Ripe | 400 | 150 | 150 |
| Overripe | 400 | 150 | 113 |
| Rotten | 400 | 150 | 150 |
| **Total** | **1,600** | **600** | **518** |

The resulting subset is approximately 58.9% training, 22.1% validation and 19.1% test. These proportions result from cleaning and class caps applied to the upstream split; the program does not randomly reassign all images to a new 60/20/20 split.

Normalization uses RGB means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`. The detailed audit is saved in [data/audit.json](data/audit.json).

## Training diagnostics and evaluation

Each epoch prints and records training/validation loss, accuracy, macro F1, and mean/max pre-clipping gradient norms. Plots are exported as PNG. Checkpoint selection uses validation macro F1. Test evaluation is performed on the chosen checkpoint for each predeclared condition; the test results do not drive training changes.

The dashboard and report show explicit heuristic thresholds for overfitting, underfitting, unstable training, convergence, vanishing gradients and large gradients. Absence of a flag is not proof of absence of a problem. Gradient norms cover trainable parameters; frozen layers are intentionally excluded. Norms are clipped to 5 and non-finite values abort training.

Evaluation includes accuracy, macro precision/recall/F1, per-class precision/recall/F1/support, confusion matrices and batched CPU inference timing. `test_predictions.csv` has true label, predicted label, score, measured brightness/edge detail and analysis for **every test image**. Suggested causes are marked as hypotheses; occlusion cannot be inferred from these simple measurements.

### Reading the diagnostics

| Indicator | Implemented rule |
|---|---|
| Overfitting | Final training/validation accuracy gap exceeds 0.10, or validation loss rises more than 15% above its minimum while training loss decreases |
| Underfitting | Final training and validation accuracies are both below 0.70 |
| Unstable training | An absolute consecutive validation-loss change exceeds 0.30 |
| Convergence | Last three validation losses span less than 0.025, with at least four recorded epochs |
| Vanishing gradients | Mean gradient norm remains below `1e-6` for the last three epochs |
| Large gradients | A recorded maximum pre-clipping norm exceeds 100; non-finite gradients abort training |

In the saved run, MobileNetV3 shows the loss-divergence overfitting indicator, and ResNet18 and MobileNetV3 satisfy the plateau indicator. The baseline triggers the instability heuristic because it uses the absolute loss change, which can also flag a rapid improvement. Interpret these flags alongside the curves. No saved run triggers the vanishing- or large-gradient thresholds.

![MobileNetV3 training and validation curves](artifacts/mobilenet/curves.png)

![MobileNetV3 test confusion matrix](artifacts/mobilenet/confusion_matrix.png)

### Examples from the recorded test predictions

These are actual rows from [MobileNetV3 test predictions](artifacts/mobilenet/test_predictions.csv). Scores are rounded for readability.

| True label | Prediction | Model score | Analysis |
|---|---|---:|---|
| Overripe | Overripe | 96.64% | Correct |
| Overripe | Rotten | 86.82% | Misclassified; possible visual overlap between stages; human review required |
| Overripe | Rotten | 50.88% | Misclassified with an uncertain score; possible visual overlap; human review required |
| Ripe | Overripe | 86.98% | Misclassified; possible visual overlap between stages; human review required |

The analysis records correctness and simple image-quality indicators. It does not establish why the neural network made an individual mistake.

## HTTP API

Start the local server before calling these endpoints.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/summary` | Experiment results, classes and data audit |
| POST | `/api/predict?model=mobilenet` | Classify raw image bytes |
| GET | `/api/errors?model=mobilenet` | All test prediction records for the selected model |
| GET | `/sample/0` | First image in the held-out test manifest |

Valid model IDs are `baseline`, `cnn_low_lr`, `resnet18`, and `mobilenet`. Omitting `model` from prediction uses the selected model in the summary. Prediction accepts the image itself as the request body, rather than multipart form data.

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/predict?model=mobilenet" -H "Content-Type: image/jpeg" --data-binary "@C:\path\to\banana.jpg"
```

Replace the example image path with an existing file. The JSON response contains `label`, `confidence`, `model`, `probabilities`, `uncertain`, and a scope note. Invalid images or model IDs return HTTP 400; invalid upload sizes return HTTP 413.

## Verification

Run the checks directly; a running server and an available network port are unnecessary:

```powershell
python verify.py
```

The seven test methods use Flask's test client to exercise the application routes. They check known duplicate-group separation between splits, stored metric consistency, validation-based model selection, inference against all four checkpoints, default-model selection, invalid inputs, upload size limits, static/report/sample routes, and file-access boundaries. Predictions and confidence scores are compared with saved test outputs. These checks do not replace a manual browser review or evaluation on independently collected fruit images.

## Files

- `models.py`: shared model architectures and inference preprocessing.
- `prepare.py`: download, cleaning, deduplication, frozen splits, class audit.
- `train.py`: experiment definitions, training, checkpoint selection, evaluation and report.
- `app.py`: Flask application, API routes, file serving and inference integration.
- `web/`: HTML, CSS and JavaScript browser interface served by Flask.
- `verify.py`: split leakage, metric consistency, model-selection and HTTP inference checks.
- `refresh_diagnostics.py`: regenerate diagnostic interpretations from saved epoch histories without retraining.
- `data/manifest.csv`, `audit.json`, `rejected.csv`: provenance and preparation evidence.
- `artifacts/REPORT.md`, `comparison.csv`, `summary.json`: measured comparison and analysis.
- `artifacts/<experiment>/best.pt`, `history.csv`, `results.json`, `curves.png`, `confusion_matrix.png`, `test_predictions.csv`: trained models and evidence.

## Troubleshooting

| Problem | Action |
|---|---|
| Missing Python module | Install `requirements.txt` with the same interpreter used to start the application |
| Port 8000 is occupied | Stop the previous project server or use `python app.py --port 8001`; `verify.py` uses Flask's test client and requires no port |
| `No module named flask` | Run `python -m pip install -r requirements.txt` with the interpreter used to start the application |
| Model has not been trained | Restore the corresponding `artifacts/<experiment>/best.pt` or run dataset preparation and training |
| Dataset/sample files missing | Run `python prepare.py --download` before training or using test samples |
| Dataset checksum mismatch | Verify the source archive; do not disable checksum validation |
| Browser shows an older interface | Restart the server from this project directory and reload the browser |

The `.gitignore` excludes the raw dataset, downloaded archive and model checkpoints. A source-only clone therefore needs dataset preparation and training, or separately supplied trusted checkpoints, before inference can work.

## Planned extension to additional fruits

Additional fruits are not yet supported by the saved checkpoints. Implementing that extension requires labeled datasets for each fruit, explicit fruit/stage label mappings, suitable split grouping, compatible model output layers, retraining, evaluation per fruit, and corresponding changes to the API and interface. Changing the displayed fruit name alone would not add recognition capability.

## Practical limits

This is a single-seed research prototype on a capped public dataset, not a validated general fruit detector. Source images may share physical fruits, acquisition sessions or adjacent video frames despite deduplication; source/session IDs are unavailable. External evaluation split by physical fruit/session and repeated seeds are necessary before claiming real-world generalization. Confidence scores are uncalibrated, and a low-confidence threshold is only an interface hint. Non-banana images can still receive confident predictions. Mixed-stage bunches and multiple fruit species are outside the evaluated task. Image appearance cannot determine food safety or internal ripeness.
