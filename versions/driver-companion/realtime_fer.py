"""
realtime_fer.py
===============
调用电脑摄像头，实时运行 Driver-Companion 项目的表情识别模型。

使用方式
--------
    pip install torch torchvision opencv-python pillow numpy

    python realtime_fer.py --checkpoint /path/to/image_output/checkpoints/best.pt

可选参数
--------
    --checkpoint   模型权重路径（默认：./image_output/checkpoints/best.pt）
    --backbone     模型结构，mobilenet_v2 / mobilenet_v3_small / resnet18（默认：mobilenet_v2）
    --camera       摄像头编号（默认：0）
    --image-size   输入分辨率（默认：224）
    --conf         显示置信度阈值，低于此值显示 "?" （默认：0.3）
    --no-bar       不显示底部置信度柱状图

操作说明
--------
    Q / ESC   退出
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

# ── 情绪标签（与训练时保持一致）──────────────────────────────────────────────
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(EMOTIONS)

# 每种情绪对应的 BGR 颜色
EMOTION_COLORS = {
    "angry":   (0,   0,   220),
    "disgust": (0,   140, 0  ),
    "fear":    (160, 0,   160),
    "happy":   (0,   200, 200),
    "neutral": (180, 180, 180),
    "sad":     (200, 80,  0  ),
    "surprise":(0,   180, 255),
}

EMOJI = {
    "angry": "😠", "disgust": "🤢", "fear": "😨",
    "happy": "😄", "neutral": "😐", "sad":  "😢", "surprise": "😲",
}

# ── 模型构建（与 train_image.py 完全一致）────────────────────────────────────
def build_backbone(name: str, num_classes: int) -> nn.Module:
    name = name.lower()
    if name == "mobilenet_v2":
        m = models.mobilenet_v2(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if name == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise ValueError(f"不支持的 backbone: {name}")


def load_model(checkpoint: str, backbone: str, device: torch.device) -> nn.Module:
    model = build_backbone(backbone, NUM_CLASSES)
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到权重文件：{ckpt_path.resolve()}")

    state = torch.load(ckpt_path, map_location=device)
    # 兼容 train_image.py 保存格式（有时 state_dict 嵌在字典里）
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ── 预处理（与训练 eval_tf 完全一致）────────────────────────────────────────
def make_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ── 人脸检测（使用 OpenCV 内置 Haar 级联）───────────────────────────────────
def get_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def detect_faces(gray: np.ndarray, detector):
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return faces if len(faces) > 0 else []


# ── 推理单张人脸 ─────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_face(face_img_rgb: np.ndarray, model, transform, device) -> tuple[str, np.ndarray]:
    """返回 (情绪标签, 各类别概率数组)"""
    pil_img = Image.fromarray(face_img_rgb)
    tensor = transform(pil_img).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    label = EMOTIONS[int(probs.argmax())]
    return label, probs


# ── 绘制工具 ──────────────────────────────────────────────────────────────────
def draw_face_box(frame: np.ndarray, x, y, w, h, label: str, conf: float, show: bool):
    """在人脸周围画框和标签"""
    color = EMOTION_COLORS.get(label, (200, 200, 200))
    thickness = 2

    # 框
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

    if not show:
        text = "? (low conf)"
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        return

    # 标签背景
    text = f"{label.upper()}  {conf:.0%}"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    cv2.rectangle(frame, (x, y - th - baseline - 8), (x + tw + 8, y), color, -1)
    cv2.putText(frame, text, (x + 4, y - baseline - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


def draw_prob_bars(frame: np.ndarray, probs: np.ndarray, top_label: str, conf_thresh: float):
    """在画面底部绘制各情绪置信度横向柱状图"""
    h, w = frame.shape[:2]
    bar_h = 18
    bar_max_w = 160
    pad_x = 12
    pad_y = h - (NUM_CLASSES * (bar_h + 4)) - 10

    # 半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (pad_x - 4, pad_y - 4),
                  (pad_x + bar_max_w + 130, pad_y + NUM_CLASSES * (bar_h + 4) + 4),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, (emotion, prob) in enumerate(zip(EMOTIONS, probs)):
        y0 = pad_y + i * (bar_h + 4)
        color = EMOTION_COLORS[emotion]
        bar_w = int(prob * bar_max_w)
        is_top = (emotion == top_label)

        # 柱体
        cv2.rectangle(frame, (pad_x + 70, y0), (pad_x + 70 + bar_w, y0 + bar_h), color, -1)
        # 情绪名
        font_scale = 0.52
        cv2.putText(frame, emotion, (pad_x, y0 + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255) if not is_top else color, 1 + int(is_top))
        # 百分比
        cv2.putText(frame, f"{prob:.0%}", (pad_x + 70 + bar_max_w + 6, y0 + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (220, 220, 220), 1)


def draw_fps(frame: np.ndarray, fps: float):
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (frame.shape[1] - 120, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2)


# ── 主循环 ────────────────────────────────────────────────────────────────────
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    print(f"[INFO] 加载模型: {args.checkpoint}")
    model = load_model(args.checkpoint, args.backbone, device)
    transform = make_transform(args.image_size)
    detector = get_face_detector()
    print("[INFO] 模型加载完毕，打开摄像头...")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头（编号 {args.camera}）")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[INFO] 按 Q 或 ESC 退出")

    fps_buf = []
    t_prev = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] 无法读取摄像头帧，退出")
            break

        # FPS 计算
        t_now = time.perf_counter()
        fps_buf.append(1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now
        if len(fps_buf) > 20:
            fps_buf.pop(0)
        fps = float(np.mean(fps_buf))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray, detector)

        last_probs = None
        last_label = None

        for (x, y, w, h) in faces:
            # 稍微扩展裁剪区域，让模型看到更多上下文
            pad = int(0.15 * min(w, h))
            x1 = max(x - pad, 0)
            y1 = max(y - pad, 0)
            x2 = min(x + w + pad, frame.shape[1])
            y2 = min(y + h + pad, frame.shape[0])

            face_bgr = frame[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

            label, probs = predict_face(face_rgb, model, transform, device)
            conf = float(probs.max())
            show = conf >= args.conf

            draw_face_box(frame, x, y, w, h, label, conf, show)
            last_probs = probs
            last_label = label

        # 底部柱状图（只针对最后一张检测到的脸）
        if last_probs is not None and not args.no_bar:
            draw_prob_bars(frame, last_probs, last_label, args.conf)

        draw_fps(frame, fps)

        # 无人脸提示
        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 255), 2)

        cv2.imshow("Driver-Companion  |  Real-time FER  |  Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):  # Q or ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] 已退出")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Driver-Companion 实时表情识别（摄像头推理）"
    )
    p.add_argument("--checkpoint",
                   default="./image_output/checkpoints/best.pt",
                   help="模型权重路径")
    p.add_argument("--backbone",
                   default="mobilenet_v2",
                   choices=["mobilenet_v2", "mobilenet_v3_small", "resnet18"],
                   help="模型结构，需与训练时一致")
    p.add_argument("--camera",
                   type=int, default=0,
                   help="摄像头编号（默认 0）")
    p.add_argument("--image-size",
                   type=int, default=224,
                   help="模型输入分辨率（默认 224）")
    p.add_argument("--conf",
                   type=float, default=0.3,
                   help="置信度阈值，低于此值显示 '?' （默认 0.3）")
    p.add_argument("--no-bar",
                   action="store_true",
                   help="关闭底部置信度柱状图")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
