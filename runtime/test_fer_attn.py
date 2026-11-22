"""
test_fer_attn.py — 独立 FER 测试脚本（预校准 Blendshape Attention）
=============================================================
使用摄像头实时测试 BlendshapeAttentionModel，显示：
  - 人脸识别框
  - 当前情绪标签 + 置信度
  - 右侧各情绪概率条

用法：
    python test_fer_attn.py
    python test_fer_attn.py --checkpoint ./blendshape_attn_calibrated.pt
    python test_fer_attn.py --camera 1 --conf 0.4

按 Q 或 ESC 退出。
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from sklearn.preprocessing import StandardScaler

# ── 常量 ─────────────────────────────────────────────────────
EMOTIONS         = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES      = len(EMOTIONS)
BLENDSHAPE_COUNT = 52

EMOTION_COLORS_BGR = {
    "angry":    (0,   0,   220),
    "disgust":  (0,   140, 0  ),
    "fear":     (160, 0,   160),
    "happy":    (0,   200, 200),
    "neutral":  (180, 180, 180),
    "sad":      (200, 80,  0  ),
    "surprise": (0,   180, 255),
}

# ── 模型定义 ──────────────────────────────────────────────────
class BlendshapeAttentionModel(nn.Module):
    def __init__(self, input_dim=52, num_classes=7, dropout=0.4):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Softmax(dim=-1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        w = self.attention(x)
        x = x * w
        return self.mlp(x)


# ── 模型加载 ──────────────────────────────────────────────────
def load_model(checkpoint: str, device: torch.device):
    ckpt   = torch.load(checkpoint, map_location=device, weights_only=False)
    dropout = ckpt.get("config", {}).get("dropout", 0.4)
    model  = BlendshapeAttentionModel(BLENDSHAPE_COUNT, NUM_CLASSES, dropout=dropout)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt["scaler"]


# ── MediaPipe ────────────────────────────────────────────────
def create_landmarker(model_path: str) -> mp_vision.FaceLandmarker:
    base = mp_python.BaseOptions(model_asset_path=model_path)
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=base,
        output_face_blendshapes=True,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts)


def extract_blendshapes(result) -> np.ndarray | None:
    if not result.face_blendshapes:
        return None
    bs     = result.face_blendshapes[0]
    values = [b.score for b in bs]          # 全部 52 个（含 index 0 _neutral）
    if len(values) < BLENDSHAPE_COUNT:
        return None
    return np.array(values[:BLENDSHAPE_COUNT], dtype=np.float32)


def get_face_bbox(result, fw: int, fh: int) -> tuple | None:
    if not result.face_landmarks:
        return None
    lms = result.face_landmarks[0]
    xs  = [lm.x * fw for lm in lms]
    ys  = [lm.y * fh for lm in lms]
    x1, y1 = int(min(xs)), int(min(ys))
    x2, y2 = int(max(xs)), int(max(ys))
    pad = 12
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(fw, x2 + pad) - max(0, x1 - pad),
        min(fh, y2 + pad) - max(0, y1 - pad),
    )


# ── 推理 ─────────────────────────────────────────────────────
@torch.no_grad()
def predict(blendshapes: np.ndarray, model, scaler, device) -> tuple[str, np.ndarray]:
    scaled = scaler.transform(blendshapes.reshape(1, -1))
    tensor = torch.tensor(scaled, dtype=torch.float32).to(device)
    probs  = torch.softmax(model(tensor), dim=1).squeeze(0).cpu().numpy()
    return EMOTIONS[int(probs.argmax())], probs


# ── 绘制 ─────────────────────────────────────────────────────
def draw_bbox_label(frame, label: str, conf: float, bbox: tuple, conf_thresh: float):
    """人脸框 + 左上角情绪标签。"""
    color = EMOTION_COLORS_BGR[label]
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    if conf >= conf_thresh:
        text = f"{label.upper()}  {conf:.0%}"
        font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
        (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
        # 标签背景
        cv2.rectangle(frame, (x, y - th - bl - 10), (x + tw + 10, y), color, -1)
        cv2.putText(frame, text, (x + 5, y - bl - 5), font, scale, (255, 255, 255), thick)
    else:
        cv2.putText(frame, "? low conf", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 120, 120), 1)


def draw_prob_bars(frame, probs: np.ndarray, top_label: str):
    """右侧纵向概率条。"""
    fh, fw   = frame.shape[:2]
    bar_h    = 22
    bar_max  = 180
    margin_r = 14          # 距右边缘
    pad_y    = fh - NUM_CLASSES * (bar_h + 6) - 14
    label_w  = 82          # 文字列宽
    pct_w    = 46          # 百分比列宽
    panel_x  = fw - margin_r - bar_max - label_w - pct_w - 8
    panel_y  = pad_y - 6
    panel_w  = label_w + bar_max + pct_w + 16
    panel_h  = NUM_CLASSES * (bar_h + 6) + 10

    # 半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, (emotion, prob) in enumerate(zip(EMOTIONS, probs)):
        y0    = pad_y + i * (bar_h + 6)
        color = EMOTION_COLORS_BGR[emotion]
        is_top = (emotion == top_label)
        bw    = int(prob * bar_max)
        bar_x = panel_x + label_w + 6

        # 条形
        cv2.rectangle(frame, (bar_x, y0), (bar_x + bw, y0 + bar_h), color, -1)
        # 灰色底槽
        cv2.rectangle(frame, (bar_x, y0), (bar_x + bar_max, y0 + bar_h),
                      (60, 60, 60), 1)

        # 情绪名
        font_scale = 0.52
        weight     = 2 if is_top else 1
        txt_color  = color if is_top else (220, 220, 220)
        cv2.putText(frame, emotion,
                    (panel_x + 4, y0 + bar_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, txt_color, weight)

        # 百分比
        cv2.putText(frame, f"{prob:.0%}",
                    (bar_x + bar_max + 5, y0 + bar_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1)


def draw_fps(frame, fps: float):
    cv2.putText(frame, f"FPS {fps:.1f}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)


def draw_no_face(frame):
    cv2.putText(frame, "No face detected", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 255), 2)


# ── 主循环 ────────────────────────────────────────────────────
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FER-test] Device : {device}")

    # 加载情绪模型
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model, scaler = load_model(str(ckpt_path), device)
    print(f"[FER-test] Model  : {ckpt_path}")

    # 加载 MediaPipe landmarker
    lm_path = Path(args.landmarker)
    if not lm_path.exists():
        raise FileNotFoundError(
            f"face_landmarker.task not found: {lm_path}\n"
            "下载地址: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker"
        )
    landmarker = create_landmarker(str(lm_path))
    print(f"[FER-test] Lndmrk : {lm_path}")

    # 打开摄像头
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"[FER-test] Camera : {args.camera}  —  press Q / ESC to quit\n")

    # 平滑缓冲（5帧多数投票）
    SMOOTH = 5
    buf    = []
    t_prev = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fh, fw = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)

        blendshapes = extract_blendshapes(result)

        if blendshapes is not None:
            label, probs = predict(blendshapes, model, scaler, device)
            conf         = float(probs.max())
            bbox         = get_face_bbox(result, fw, fh)

            # 平滑
            buf.append(label)
            if len(buf) > SMOOTH:
                buf.pop(0)
            smooth_label = max(set(buf), key=buf.count) if buf else label

            if bbox:
                draw_bbox_label(frame, smooth_label, conf, bbox, args.conf)
            draw_prob_bars(frame, probs, smooth_label)
        else:
            buf.clear()
            draw_no_face(frame)

        # FPS
        t_now  = time.perf_counter()
        fps    = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now
        draw_fps(frame, fps)

        cv2.imshow("CalmWheel — FER test (blendshape_attn)  |  Q to quit", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[FER-test] Stopped.")


# ── CLI ───────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="CalmWheel FER 独立测试（预校准 Blendshape Attention）")
    p.add_argument("--checkpoint",
                   default="./blendshape_attn_calibrated.pt",
                   help="情绪模型路径")
    p.add_argument("--landmarker",
                   default="./face_landmarker.task",
                   help="MediaPipe face_landmarker.task 路径")
    p.add_argument("--camera",
                   type=int, default=0,
                   help="摄像头编号（默认 0）")
    p.add_argument("--conf",
                   type=float, default=0.3,
                   help="显示标签的最低置信度阈值（默认 0.3）")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
