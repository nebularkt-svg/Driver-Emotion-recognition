"""
realtime_fer.py  — Blendshape Attention Model
==============================================
Uses MediaPipe FaceLandmarker to extract 52 blendshape coefficients per frame,
then classifies emotion with the attention+MLP model (93.6% accuracy).

Inputs:  camera frames
Outputs: on_emotion(emotion, confidence) callback
         on_frame(annotated_bgr) callback → ui.py camera panel

Usage (standalone):
    python realtime_fer.py --checkpoint ./blendshape_attn_calibrated.pt
"""

import argparse
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# MediaPipe
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

EMOTIONS    = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(EMOTIONS)
BLENDSHAPE_COUNT = 52

EMOTION_COLORS_BGR = {
    "angry":   (0,   0,   220),
    "disgust": (0,   140, 0  ),
    "fear":    (160, 0,   160),
    "happy":   (0,   200, 200),
    "neutral": (180, 180, 180),
    "sad":     (200, 80,  0  ),
    "surprise":(0,   180, 255),
}


# ── Model definition (must match training) ───────────────────
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


class BlendshapeMLP(nn.Module):
    def __init__(self, input_dim=52, num_classes=7, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
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
        return self.net(x)


def _calibrate_bn(model: nn.Module, scaler, train_npz_path: str, device: torch.device,
                   epochs: int = 100, batch: int = 16) -> None:
    """Fix corrupted BatchNorm running stats by re-estimating them on training data.

    The saved checkpoints have near-zero running_var in their BN layers, which
    causes eval-mode inference to collapse all predictions to a single class
    ("sad"). Re-running training-set samples in train mode reconstructs the
    correct running statistics without touching the learnable parameters.
    """
    import numpy as _np
    data = _np.load(train_npz_path)
    x = torch.tensor(scaler.transform(data["blendshapes"].astype(_np.float32)),
                     dtype=torch.float32, device=device)

    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.reset_running_stats()
            m.momentum = 0.01

    model.train()
    for p in model.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        for _ in range(epochs):
            idx = torch.randperm(len(x), device=device)
            for i in range(0, len(x), batch):
                b = x[idx[i:i + batch]]
                if b.shape[0] >= 2:
                    model(b)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    print("[FER] BatchNorm stats calibrated.")


def load_model(checkpoint: str, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    dropout = ckpt.get("config", {}).get("dropout", 0.4)
    model_type = ckpt.get("config", {}).get("model", "attention")
    if model_type == "mlp":
        model = BlendshapeMLP(BLENDSHAPE_COUNT, NUM_CLASSES, dropout=dropout)
    else:
        model = BlendshapeAttentionModel(BLENDSHAPE_COUNT, NUM_CLASSES, dropout=dropout)
    model.load_state_dict(ckpt["model"])
    model.to(device)

    # Re-calibrate BN running stats if a training cache is available.
    # Saved checkpoints often have corrupted running_var (near-zero), which
    # causes eval-mode BN to collapse every prediction to the same class.
    _here = Path(__file__).resolve().parent
    _TRAIN_CACHE_CANDIDATES = [
        str(_here / "cache/ckplus_static/ckplus_train.npz"),
        str(_here.parent.parent / "PycharmProjects/Driver-Companion/emotion-reco/cache/ckplus_static/ckplus_train.npz"),
        str(_here.parent / "Driver-Companion/emotion-reco/cache/ckplus_static/ckplus_train.npz"),
    ]
    scaler = ckpt["scaler"]
    calibrated = False
    for candidate in _TRAIN_CACHE_CANDIDATES:
        if Path(candidate).exists():
            _calibrate_bn(model, scaler, candidate, device)
            calibrated = True
            break
    if not calibrated:
        print("[FER] Warning: training cache not found — BN stats not recalibrated. "
              "Use a pre-calibrated checkpoint to avoid class-collapse predictions.")

    model.eval()
    return model, scaler


# ── MediaPipe face landmarker setup ──────────────────────────
def create_face_landmarker(model_path: str):
    """
    Create MediaPipe FaceLandmarker for blendshape extraction.
    Requires face_landmarker.task — download from:
    https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
    """
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def extract_blendshapes(result) -> Optional[np.ndarray]:
    """Extract 52 blendshape floats from MediaPipe result."""
    if not result.face_blendshapes:
        return None
    blendshapes = result.face_blendshapes[0]
    # Use all 52 blendshapes (indices 0..51, including _neutral at index 0).
    # This matches the training extraction in Dataset/Dataset.py which does:
    #   coeffs = np.array([c.score for c in bs], dtype=np.float32)
    # The original code skipped index 0 and tried blendshapes[1:53], but
    # MediaPipe only returns 52 items (0..51), so [1:53] yields 51 values,
    # triggering the length check below and silently returning None every frame.
    values = [bs.score for bs in blendshapes]
    if len(values) < BLENDSHAPE_COUNT:
        return None
    return np.array(values[:BLENDSHAPE_COUNT], dtype=np.float32)


# ── Inference ────────────────────────────────────────────────
@torch.no_grad()
def predict(blendshapes: np.ndarray, model, scaler, device) -> tuple[str, np.ndarray]:
    scaled = scaler.transform(blendshapes.reshape(1, -1))
    tensor = torch.tensor(scaled, dtype=torch.float32).to(device)
    logits = model(tensor)
    probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    label  = EMOTIONS[int(probs.argmax())]
    return label, probs


# ── Frame annotation ─────────────────────────────────────────
def annotate_frame(frame, label, probs, conf, conf_thresh, show_bars, face_bbox=None):
    color = EMOTION_COLORS_BGR.get(label, (200, 200, 200))

    # Face bounding box if available
    if face_bbox is not None:
        x, y, w, h = face_bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if conf >= conf_thresh:
            text = f"{label.upper()}  {conf:.0%}"
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.rectangle(frame, (x, y - th - bl - 8), (x + tw + 8, y), color, -1)
            cv2.putText(frame, text, (x + 4, y - bl - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "? (low conf)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
    else:
        # No bbox — show label top-left
        if conf >= conf_thresh:
            cv2.putText(frame, f"{label.upper()} {conf:.0%}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if show_bars and probs is not None:
        _draw_bars(frame, probs, label)

    return frame


def _draw_bars(frame, probs, top_label):
    fh, fw  = frame.shape[:2]
    bar_h   = 18
    bar_max = 160
    pad_x   = 12
    pad_y   = fh - (NUM_CLASSES * (bar_h + 4)) - 10
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (pad_x - 4, pad_y - 4),
                  (pad_x + bar_max + 130, pad_y + NUM_CLASSES * (bar_h + 4) + 4),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, (emotion, prob) in enumerate(zip(EMOTIONS, probs)):
        y0     = pad_y + i * (bar_h + 4)
        color  = EMOTION_COLORS_BGR[emotion]
        bw     = int(prob * bar_max)
        is_top = (emotion == top_label)
        cv2.rectangle(frame, (pad_x + 70, y0), (pad_x + 70 + bw, y0 + bar_h), color, -1)
        cv2.putText(frame, emotion, (pad_x, y0 + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255, 255, 255) if not is_top else color, 1 + int(is_top))
        cv2.putText(frame, f"{prob:.0%}",
                    (pad_x + 70 + bar_max + 6, y0 + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)


# ── Face bbox from landmarks ──────────────────────────────────
def get_face_bbox(result, frame_w, frame_h):
    if not result.face_landmarks:
        return None
    lms = result.face_landmarks[0]
    xs  = [lm.x * frame_w for lm in lms]
    ys  = [lm.y * frame_h for lm in lms]
    x1, y1 = int(min(xs)), int(min(ys))
    x2, y2 = int(max(xs)), int(max(ys))
    pad = 10
    return (max(0, x1 - pad), max(0, y1 - pad),
            min(frame_w, x2 + pad) - max(0, x1 - pad),
            min(frame_h, y2 + pad) - max(0, y1 - pad))


# ── Main loop ─────────────────────────────────────────────────
def run(args,
        on_emotion: Optional[Callable[[str, float], None]] = None,
        on_frame:   Optional[Callable[[np.ndarray], None]] = None):

    SMOOTH_FRAMES     = 5
    CALLBACK_COOLDOWN = 15.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FER] Device: {device}")
    print(f"[FER] Loading blendshape model: {args.checkpoint}")

    model, scaler = load_model(args.checkpoint, device)
    print("[FER] Model ready.")

    # MediaPipe landmarker
    lm_model_path = getattr(args, "landmarker", "face_landmarker.task")
    if not Path(lm_model_path).exists():
        raise FileNotFoundError(
            f"MediaPipe face_landmarker.task not found at '{lm_model_path}'.\n"
            "Download from: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker"
        )
    landmarker = create_face_landmarker(lm_model_path)
    print("[FER] MediaPipe landmarker ready.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    emotion_buf     = []
    last_callback_t = 0.0
    last_sent       = None

    print("[FER] Running. Press Q or ESC to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fh, fw = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result     = landmarker.detect(mp_img)
        blendshapes = extract_blendshapes(result)

        label, probs, conf, bbox = None, None, 0.0, None

        if blendshapes is not None:
            label, probs = predict(blendshapes, model, scaler, device)
            conf         = float(probs.max())
            bbox         = get_face_bbox(result, fw, fh)

        # Annotate
        annotated = annotate_frame(
            frame.copy(), label or "neutral", probs,
            conf, args.conf, not args.no_bar, bbox
        )
        if label is None:
            cv2.putText(annotated, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 255), 2)

        # Send frame to UI
        if on_frame is not None:
            on_frame(annotated)

        # Smooth + callback
        if label is not None and on_emotion is not None:
            emotion_buf.append(label)
            if len(emotion_buf) > SMOOTH_FRAMES:
                emotion_buf.pop(0)

            if len(emotion_buf) == SMOOTH_FRAMES:
                dominant = max(set(emotion_buf), key=emotion_buf.count)
                count    = emotion_buf.count(dominant)
                now      = time.perf_counter()

                if (count >= SMOOTH_FRAMES * 0.6 and
                        (dominant != last_sent or
                         now - last_callback_t > CALLBACK_COOLDOWN)):
                    conf_val = float(probs[EMOTIONS.index(dominant)])
                    on_emotion(dominant, conf_val)
                    last_sent       = dominant
                    last_callback_t = now
        else:
            emotion_buf.clear()

        # Standalone mode
        if on_frame is None:
            cv2.imshow("FER — Blendshape Attention | Q to quit", annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break

    cap.release()
    if on_frame is None:
        cv2.destroyAllWindows()
    print("[FER] Stopped.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default="./blendshape_attn_calibrated.pt")
    p.add_argument("--landmarker",  default="face_landmarker.task",
                   help="Path to MediaPipe face_landmarker.task")
    p.add_argument("--camera",      type=int,   default=0)
    p.add_argument("--conf",        type=float, default=0.3)
    p.add_argument("--no-bar",      action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
