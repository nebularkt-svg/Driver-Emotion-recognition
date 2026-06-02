# Driver-Emotion-recognition

Real-time, seven-class driver facial expression recognition for CalmWheel, an emotion-aware driving companion. Uses MediaPipe blendshapes and PyTorch, with archived model iterations, training checkpoints, and evaluation results.

CalmWheel 的摄像头面部表情识别模块，以及课程项目中保留的识别版本、训练记录和模型权重。

## 运行结题演示版

安装 Python 3.11 和 Git LFS。在仓库根目录用 PowerShell 执行：

```powershell
git lfs pull
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r runtime/requirements.txt
Push-Location runtime
..\.venv\Scripts\python.exe realtime_fer.py --checkpoint blendshape_attn_calibrated.pt --landmarker face_landmarker.task --camera 0
Pop-Location
```

结题演示快照中的实时入口默认加载 `blendshape_attn_calibrated.pt`。MediaPipe Face Landmarker 提取 blendshape，校准后的 Attention 模型分类，OpenCV 显示摄像头结果。按 Q 退出。历史演示代码会在找不到训练缓存时提示未重新校准，运行命令已指定预校准权重。

## 版本与实验

| 目录 | 内容 |
| --- | --- |
| [runtime](runtime/) | 结题演示使用的实时识别、预校准权重和 MediaPipe 模型 |
| [versions/cnn-yolov8](versions/cnn-yolov8/) | SHRI 导出的早期 CNN 和 YOLOv8 人脸检测、训练脚本及存档权重 |
| [versions/driver-companion](versions/driver-companion/) | Driver Companion 快照，含基准测试与图像识别 |
| [versions/recognition-snapshot](versions/recognition-snapshot/) | 独立识别快照，含实时图像与 blendshape 入口，以及训练和数据集脚本 |
| [versions/calmwheel](versions/calmwheel/) 和 [versions/fer-bridge](versions/fer-bridge/) | 产品整合阶段的摄像头识别接口、语音识别辅助脚本和训练快照 |
| [versions/mesh](versions/mesh/) | 结题演示前的 MediaPipe Face Mesh 与 Blendshape Attention 快照 |
| [versions/final-demo-image](versions/final-demo-image/) | 结题演示目录中仍保留的图像模型权重 |
| [versions/archive-variants](versions/archive-variants/) | ZIP 中与已解压副本有内容差异的代码和权重 |
| [experiments/runs](experiments/runs/) | CK+/RAVDESS 的交叉验证权重、标签和 scaler，以及 EmoNet、几何特征、Blendshape MLP 和 Attention 的训练日志、指标及图表 |

版本目录名依据工作区来源命名。历史模型和未成为结题默认入口的实验均保留，原始训练报告中的指标沿用各自的数据划分和评估方法。历史运行环境和第三方数据集需按相应版本的脚本及说明准备。识别仓库只收录识别侧代码和产物，历史整合说明中的 LLM、界面及硬件启动步骤需要原集成项目。ConvLSTM 交叉验证目录包含训练配置、指标和权重，工作区中未找到对应的训练源码。其余训练脚本可在各版本目录的 TrainVla 中查阅。可在独立识别快照下运行 `python TrainVla/run_all_benchmarks.py --help` 查看基准入口。

[模型清单](model-manifest.csv)记录每个二进制模型的目标路径、包内位置、大小、SHA-256 和原始来源。数据集照片、训练缓存、密钥、个人演示视频和无关文件未收录。公开发布前请按原来源确认第三方模型和代码的再分发许可。

## 推送到 GitHub

创建空的 GitHub 仓库，在本目录替换远程地址并执行：

```powershell
git remote add origin https://github.com/OWNER/REPO.git
git push -u origin main
```

常规模型随 Git LFS 推送，去重后约 2.56 GiB。超过 500 MiB 的 CK+ 122 和 RAVDESS 122 权重放在本地 `release-assets/`，共 12 个文件、约 9.05 GiB，避免这组大文件占用 Git LFS 额度。推送后，在已登录 GitHub CLI 的环境中发布该目录内的权重：

```powershell
$recognitionAssets = Get-ChildItem -LiteralPath release-assets -File | Select-Object -ExpandProperty FullName
gh release create models-v1 @recognitionAssets --target main --title "Archived training checkpoints"
```

要恢复这些权重，可在克隆后的仓库运行：

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

[Git LFS 额度](https://docs.github.com/en/billing/concepts/product-billing/git-lfs) · [GitHub Release 附件限制](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
