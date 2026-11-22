# fer_bridge.py
import sys
import os
import threading
import time
import numpy as np
import torch
import torch.nn as nn

UNIFIED_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = 7
GEO_DIM     = 17

_LM = {
    "mouth_left": 61,   "mouth_right":  291,
    "mouth_top":  13,   "mouth_bottom": 14,
    "left_eye_inner":  133, "left_eye_outer":  33,
    "left_eye_top":    159, "left_eye_bottom": 145,
    "right_eye_inner": 362, "right_eye_outer": 263,
    "right_eye_top":   386, "right_eye_bottom":374,
    "left_brow_inner": 107, "left_brow_outer":  46,
    "right_brow_inner":336, "right_brow_outer": 276,
    "nose_tip": 4, "nose_base": 2,
    "chin": 152,   "forehead":  10,
    "left_cheek": 234, "right_cheek": 454,
}

_DISTANCE_PAIRS = [
    ("mouth_left",  "mouth_right"),
    ("mouth_top",   "mouth_bottom"),
    ("left_eye_inner",  "left_eye_outer"),
    ("right_eye_inner", "right_eye_outer"),
    ("left_eye_top",    "left_eye_bottom"),
    ("right_eye_top",   "right_eye_bottom"),
    ("left_brow_inner", "left_brow_outer"),
    ("right_brow_inner","right_brow_outer"),
    ("nose_tip",    "mouth_top"),
    ("chin",        "forehead"),
    ("left_cheek",  "right_cheek"),
    ("left_eye_top","left_brow_inner"),
    ("right_eye_top","right_brow_inner"),
]


def extract_geometric_features(landmarks: np.ndarray) -> np.ndarray:
    feats = []
    for a, b in _DISTANCE_PAIRS:
        pa = landmarks[_LM[a], :2]
        pb = landmarks[_LM[b], :2]
        feats.append(float(np.linalg.norm(pa - pb)))
    face_h = feats[9] if feats[9] > 1e-6 else 1.0
    feats  = [f / face_h for f in feats]
    mouth_w = feats[0]; mouth_h = feats[1]
    eye_w_l = feats[2]; eye_h_l = feats[4]
    eye_w_r = feats[3]; eye_h_r = feats[5]
    feats.append(mouth_h / (mouth_w + 1e-6))
    feats.append(eye_h_l / (eye_w_l + 1e-6))
    feats.append(eye_h_r / (eye_w_r + 1e-6))
    feats.append(abs(eye_w_l - eye_w_r) / (eye_w_l + eye_w_r + 1e-6))
    return np.array(feats, dtype=np.float32)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True),
        )
    def forward(self, x):
        return self.block(x)


class EmoNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.4):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlock(3, 32), ConvBlock(32, 64),
            ConvBlock(64, 128), ConvBlock(128, 256),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(GEO_DIM, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 32),      nn.ReLU(inplace=True),
        )
        fused_dim = 256 + 32
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, fused_dim), nn.ReLU(inplace=True),
            nn.Linear(fused_dim, fused_dim), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),        nn.ReLU(inplace=True),
        )
        self.emotion_out = nn.Linear(64, num_classes)

    def forward(self, grid, geo):
        z = torch.cat([self.cnn(grid), self.mlp(geo)], dim=1)
        z = self.fusion(z)
        return self.emotion_out(self.head(z))


def _lm_to_grid(landmarks: np.ndarray) -> np.ndarray:
    flat   = landmarks.flatten()
    padded = np.zeros(22*22*3, dtype=np.float32)
    padded[:len(flat)] = flat
    return padded.reshape(3, 22, 22)


def _find_task_file() -> str:
    """Search common locations for face_landmarker.task."""
    base = os.path.dirname(__file__)
    candidates = [
        os.path.join(base, "face_landmarker.task"),
        os.path.join(base, "Dataset", "face_landmarker.task"),
        os.path.join(base, "..", "Dataset", "face_landmarker.task"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError(
        "face_landmarker.task not found. "
        "Copy it from the Dataset folder into the project root."
    )


class FERBridge:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.checkpoint_path  = checkpoint_path
        self.device           = torch.device(device)
        self._emotion         = "neutral"
        self._confidence      = 0.0
        self._lock            = threading.Lock()
        self._running         = False
        self._model           = None
        self._face_landmarker = None

    def _load_model(self):
        model = EmoNet().to(self.device)

        # Fix: weights_only=False needed because checkpoint contains sklearn objects
        ckpt = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False      # ← fixes the sklearn.StandardScaler error
        )

        # Handle all checkpoint formats
        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]           # ← this project's format
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and any(
            k.startswith(("cnn.", "mlp.", "fusion.", "head.", "emotion_out."))
            for k in ckpt.keys()
        ):
            state = ckpt
        else:
            state = ckpt

        model.load_state_dict(state, strict=False)
        model.eval()
        print(f"[FER] Model loaded from {self.checkpoint_path}")
        return model

    def _load_mediapipe(self):
        import mediapipe as mp
        BaseOptions          = mp.tasks.BaseOptions
        FaceLandmarker       = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode    = mp.tasks.vision.RunningMode

        task_path = _find_task_file()
        options   = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=task_path),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
        )
        print(f"[FER] MediaPipe loaded from {task_path}")
        return FaceLandmarker.create_from_options(options)

    def _predict(self, frame_rgb: np.ndarray) -> tuple[str, float]:
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result   = self._face_landmarker.detect(mp_image)

        if not result.face_landmarks:
            return self._emotion, self._confidence

        lm_raw    = result.face_landmarks[0]
        landmarks = np.array([[p.x, p.y, p.z] for p in lm_raw], dtype=np.float32)

        grid = torch.tensor(
            _lm_to_grid(landmarks), dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        geo  = torch.tensor(
            extract_geometric_features(landmarks), dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self._model(grid, geo)
            probs  = torch.softmax(logits, dim=1)[0]
            idx    = int(probs.argmax())
            conf   = float(probs[idx])

        return UNIFIED_EMOTIONS[idx], conf

    def _loop(self):
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[FER] No camera — FER disabled")
            return

        print("[FER] Camera running")
        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                emotion, conf = self._predict(frame_rgb)
                with self._lock:
                    self._emotion    = emotion
                    self._confidence = conf
            except Exception as e:
                print(f"[FER] Predict error: {e}")

            time.sleep(0.1)

        cap.release()

    def start(self):
        try:
            self._model           = self._load_model()
            self._face_landmarker = self._load_mediapipe()
        except Exception as e:
            print(f"[FER] Failed to load: {e}")
            return

        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        print("[FER] Started")

    def stop(self):
        self._running = False

    def get_emotion(self) -> tuple[str, float]:
        with self._lock:
            return self._emotion, self._confidence
