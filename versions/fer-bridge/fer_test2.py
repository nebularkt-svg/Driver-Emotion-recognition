# fer_test2.py
import os, sys, numpy as np, torch

BASE = os.path.dirname(__file__)
CHECKPOINT = os.path.join(BASE, "blendshape_attn_output", "checkpoints", "best.pt")
TASK = os.path.join(BASE, "Dataset", "face_landmarker.task")

print("="*50)

# 1. Load checkpoint
print("\n1. Loading checkpoint...")
ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
print(f"   keys: {list(ckpt.keys())}")

# 2. Load model using ckpt['model'] directly
print("\n2. Loading into EmoNet using ckpt['model']...")
sys.path.insert(0, BASE)
from fer_bridge import EmoNet, extract_geometric_features, _lm_to_grid, UNIFIED_EMOTIONS

model = EmoNet()
state = ckpt['model']   # correct key
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"   missing={len(missing)}, unexpected={len(unexpected)}")
if missing:    print(f"   missing: {missing[:3]}")
if unexpected: print(f"   unexpected: {list(unexpected)[:3]}")
model.eval()
print("   Model ready")

# 3. Camera frame
print("\n3. Grabbing camera frame...")
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
print(f"   frame shape: {frame_rgb.shape}")

# 4. MediaPipe
print("\n4. Running MediaPipe...")
import mediapipe as mp
options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=TASK),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_faces=1,
)
landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb))
print(f"   Faces detected: {len(result.face_landmarks)}")
if not result.face_landmarks:
    print("   No face — sit in front of camera and retry")
    sys.exit(0)

lm = np.array([[p.x,p.y,p.z] for p in result.face_landmarks[0]], dtype=np.float32)

# 5. Inference
print("\n5. Running inference...")
grid = torch.tensor(_lm_to_grid(lm)).unsqueeze(0)
geo  = torch.tensor(extract_geometric_features(lm)).unsqueeze(0)
with torch.no_grad():
    probs = torch.softmax(model(grid, geo), dim=1)[0]
    idx   = int(probs.argmax())
    conf  = float(probs[idx])

print(f"   Predicted: {UNIFIED_EMOTIONS[idx]} ({conf:.1%})")
print(f"   All probs: { {e: f'{p:.1%}' for e,p in zip(UNIFIED_EMOTIONS, probs.tolist())} }")

if conf > 0.3:
    print("\n✓ Model working correctly!")
else:
    print("\n⚠ Probabilities still uniform — model weights may not match architecture")
