# CalmWheel Driver Emotion Recognition

A real-time webcam demo that classifies seven facial expressions for **CalmWheel**, a COMPSYS 731 driver companion prototype. Its inference pipeline uses MediaPipe Face Landmarker, 52 facial blendshape scores, and a calibrated PyTorch attention classifier. The archive also preserves the training experiments and earlier model designs.

[Run the webcam demo](#run-the-webcam-demo) · [Explore model versions](#versions-and-experiments) · [Download archived weights](#model-downloads)

## CalmWheel project

| Component | Start here |
| --- | --- |
| Webcam inference | [Runtime code and demo weights](runtime/) |
| Model comparison | [Training methods, metrics, and curves](https://github.com/nebularkt-svg/Driver-Emotion-Model-Benchmarks) |
| Dialogue and voice | [Emotion-aware LLM module](https://github.com/nebularkt-svg/Driver-Companion-LLM) |

## Recognition pipeline

The demo uses MediaPipe Face Landmarker to extract **52 facial blendshape scores** from webcam frames. A calibrated PyTorch Blendshape Attention classifier scores seven categories, and OpenCV displays the predicted class and confidence.

**Classes:** angry, disgust, fear, happy, neutral, sad, and surprise.

## Run the webcam demo

Use Python 3.11, Git LFS, and a webcam. The PowerShell commands below download only the demo's LFS files, so you can run it without fetching the full training archive.

```powershell
$env:GIT_LFS_SKIP_SMUDGE = "1"
git clone https://github.com/nebularkt-svg/Driver-Emotion-recognition.git
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
cd Driver-Emotion-recognition
git lfs pull --include="runtime/**" --exclude=""

python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r runtime/requirements.txt
Push-Location runtime
..\.venv\Scripts\python.exe realtime_fer.py --camera 0
Pop-Location
```

Use `--camera 1` to select a second webcam. Press **Q** or **Esc** to exit. The demo ships with a pre-calibrated checkpoint and may print a message about skipping recalibration when no training cache is present.

## Versions and experiments

| Location | Contents |
| --- | --- |
| [Final demo](runtime/) | Webcam inference, pre-calibrated attention weights, and the MediaPipe model |
| [Early CNN and YOLOv8](versions/cnn-yolov8/) | Image classifier and face detector prototypes, training scripts, and checkpoints |
| [Driver Companion](versions/driver-companion/) and [recognition snapshot](versions/recognition-snapshot/) | Standalone inference, dataset preparation, training, and benchmark scripts |
| [CalmWheel](versions/calmwheel/), [FER bridge](versions/fer-bridge/), and [mesh](versions/mesh/) | Recognition integration snapshots and MediaPipe iterations |
| [Demo image weights](versions/final-demo-image/) and [ZIP variants](versions/archive-variants/) | Alternate image checkpoints and variants recovered from source archives |
| [Training runs](experiments/runs/) | CK+ and RAVDESS cross-validation, EmoNet, geometric features, and Blendshape MLP and Attention results |

Historical results use the original data splits and evaluation protocols for each run. Some snapshots depend on the original companion application's UI or speech integration. The ConvLSTM archive includes configurations, metrics, and weights. Its training source was unavailable in the workspace. Dataset images and training caches are excluded.

For training and benchmark entry points, see [TrainVla](versions/recognition-snapshot/TrainVla/) and the [dataset preparation scripts](versions/recognition-snapshot/Dataset/).

## Model downloads

| Files | Access |
| --- | --- |
| Demo weights and most historical checkpoints | Git LFS. Run `git lfs pull` to fetch the complete LFS collection, about 2.56 GiB. |
| CK+ 122 and RAVDESS 122 ConvLSTM checkpoints | [Release models-v1](https://github.com/nebularkt-svg/Driver-Emotion-recognition/releases/tag/models-v1). Each experiment includes five folds and its best checkpoint. The 12 files total about 9.05 GiB. |

The [model manifest](model-manifest.csv) lists each file's source, destination, size, and SHA-256 checksum. To download and restore the Release checkpoints to their experiment directories, run the following from the repository root with GitHub CLI installed.

```powershell
gh release download models-v1 --dir release-assets
Import-Csv model-manifest.csv | Where-Object package_path -Like 'release-assets/*' | ForEach-Object {
    $weight = $_
    if ((Get-FileHash -LiteralPath $weight.package_path -Algorithm SHA256).Hash -ne $weight.sha256) {
        throw "SHA-256 mismatch for $($weight.package_path)"
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $weight.model_path) | Out-Null
    Copy-Item -LiteralPath $weight.package_path -Destination $weight.model_path
}
```
