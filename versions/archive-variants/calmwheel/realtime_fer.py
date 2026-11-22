"""
realtime_fer.py  (integrated version)
Runs FER inference and feeds results into CalmWheel via callback.
Camera display is handled externally by ui.py — this module
returns annotated frames instead of showing its own window.
"""

import argparse
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

EMOTIONS    = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(EMOTIONS)

EMOTION_COLORS_BGR = {
    "angry":   (0,   0,   220),
    "disgust": (0,   140, 0  ),
    "fear":    (160, 0,   160),
    "happy":   (0,   200, 200),
    "neutral": (180, 180, 180),
    "sad":     (200, 80,  0  ),
    "surprise":(0,   180, 255),
}


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
    raise ValueError(f"Unsupported backbone: {name}")


def load_model(checkpoint: str, backbone: str, device: torch.device) -> nn.Module:
    model     = build_backbone(backbone, NUM_CLASSES)
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path.resolve()}")
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def make_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_face_detector():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


def detect_faces(gray, detector):
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return faces if len(faces) > 0 else []


@torch.no_grad()
def predict_face(face_rgb, model, transform, device):
    pil_img = Image.fromarray(face_rgb)
    tensor  = transform(pil_img).unsqueeze(0).to(device)
    probs   = torch.softmax(model(tensor), dim=1).squeeze(0).cpu().numpy()
    return EMOTIONS[int(probs.argmax())], probs


def annotate_frame(frame, faces_data, conf_thresh, show_bars):
    """
    Draw face boxes, labels, confidence bars onto frame in-place.
    faces_data: list of (x,y,w,h, label, probs)
    Returns annotated frame.
    """
    last_probs = None
    last_label = None

    for (x, y, w, h, label, probs) in faces_data:
        conf  = float(probs.max())
        show  = conf >= conf_thresh
        color = EMOTION_COLORS_BGR.get(label, (200, 200, 200))

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if show:
            text = f"{label.upper()}  {conf:.0%}"
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            cv2.rectangle(frame, (x, y - th - bl - 8), (x + tw + 8, y), color, -1)
            cv2.putText(frame, text, (x + 4, y - bl - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "? (low conf)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        last_probs = probs
        last_label = label

    if last_probs is not None and show_bars:
        _draw_bars(frame, last_probs, last_label)

    if not faces_data:
        cv2.putText(frame, "No face detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 255), 2)

    return frame


def _draw_bars(frame, probs, top_label):
    h, w    = frame.shape[:2]
    bar_h   = 18
    bar_max = 160
    pad_x   = 12
    pad_y   = h - (NUM_CLASSES * (bar_h + 4)) - 10
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


def run(args,
        on_emotion: Optional[Callable[[str, float], None]] = None,
        on_frame:   Optional[Callable[[np.ndarray], None]] = None):
    """
    Main FER loop.

    on_emotion(emotion, confidence): called when a stable emotion is detected.
    on_frame(annotated_frame):       called every frame with the annotated BGR image,
                                     so ui.py can display it instead of a separate window.
    """
    SMOOTH_FRAMES     = 5
    CALLBACK_COOLDOWN = 5.0    # same emotion won't re-trigger for 5 seconds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FER] Device: {device}")
    print(f"[FER] Loading: {args.checkpoint}")

    model     = load_model(args.checkpoint, args.backbone, device)
    transform = make_transform(args.image_size)
    detector  = get_face_detector()
    print("[FER] Ready.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    emotion_buf     = []
    last_callback_t = 0.0
    last_sent       = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_raw  = detect_faces(gray, detector)
        faces_data = []

        for (x, y, w, h) in faces_raw:
            pad = int(0.15 * min(w, h))
            x1  = max(x - pad, 0);  y1 = max(y - pad, 0)
            x2  = min(x + w + pad, frame.shape[1])
            y2  = min(y + h + pad, frame.shape[0])
            face_rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            label, probs = predict_face(face_rgb, model, transform, device)
            faces_data.append((x, y, w, h, label, probs))

        # Annotate frame (boxes + bars)
        annotated = annotate_frame(
            frame.copy(), faces_data, args.conf, not args.no_bar
        )

        # Send annotated frame to UI
        if on_frame is not None:
            on_frame(annotated)

        # Smooth + callback
        if faces_data and on_emotion is not None:
            top_label = faces_data[0][4]
            emotion_buf.append(top_label)
            if len(emotion_buf) > SMOOTH_FRAMES:
                emotion_buf.pop(0)

            if len(emotion_buf) == SMOOTH_FRAMES:
                dominant = max(set(emotion_buf), key=emotion_buf.count)
                count    = emotion_buf.count(dominant)
                now      = time.perf_counter()

                if (count >= SMOOTH_FRAMES * 0.6 and
                        (dominant != last_sent or
                         now - last_callback_t > CALLBACK_COOLDOWN)):
                    conf_val = float(faces_data[0][5][EMOTIONS.index(dominant)])
                    on_emotion(dominant, conf_val)
                    last_sent       = dominant
                    last_callback_t = now
        elif not faces_data:
            emotion_buf.clear()

        # Standalone mode: show own window
        if on_frame is None:
            cv2.imshow("FER", annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break

    cap.release()
    if on_frame is None:
        cv2.destroyAllWindows()
    print("[FER] Stopped.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="./image_output/checkpoints/best.pt")
    p.add_argument("--backbone",   default="mobilenet_v2",
                   choices=["mobilenet_v2", "mobilenet_v3_small", "resnet18"])
    p.add_argument("--camera",     type=int,   default=0)
    p.add_argument("--image-size", type=int,   default=224)
    p.add_argument("--conf",       type=float, default=0.3)
    p.add_argument("--no-bar",     action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
