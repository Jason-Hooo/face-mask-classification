
import os
import sys
import time
import urllib.request
import cv2
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.model import EfficientNet
from src.dataset import test_transform

"""Run real-time face mask classification with MediaPipe face detection and an EfficientNet classifier."""

EFFICIENTNET_MODEL_PATH = os.path.join(BASE_DIR, 'weights', 'trained_model_parameters.pth')
BLAZEFACE_MODEL_PATH = os.path.join(BASE_DIR, 'weights', 'blaze_face_short_range.tflite')
BLAZEFACE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite'

PAD_W_RATIO = 0.3
PAD_H_TOP_RATIO = 0.5
PAD_H_BOTTOM_RATIO = 0.1

def ensure_model_exists(path, url) -> None:
    if not os.path.exists(path):
        print(f"Model file '{path}' not found. Downloading from network...")
        try:
            urllib.request.urlretrieve(url, path)
            print("Download completed successfully!")
        except Exception as e:
            print(f"Download failed. Error: {e}")
            raise SystemExit(1)

def draw_hud(img, device_name):
    """Draw a semi-transparent status information bar (HUD) at the bottom of the screen"""
    h, w = img.shape[:2]
    hud_h = 60
    overlay = img.copy()

    cv2.rectangle(overlay, (0, h - hud_h), (w, h), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    status_text = f"Hardware: {str(device_name).upper()} | Press 'Q' to Exit"
    cv2.putText(img, status_text, (20, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 1, cv2.LINE_AA)

ensure_model_exists(BLAZEFACE_MODEL_PATH, BLAZEFACE_MODEL_URL)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

LABELS = ["Mask on chin", "Mask not covering nose", "Mask properly worn", "No mask"]
COLORS = [
    (0, 165, 255),  # Orange: Mask on chin
    (0, 165, 255),  # Orange: Mask not covering nose
    (0, 255, 0),    # Green: Mask properly worn
    (0, 0, 255)     # Red: No mask
]

def main():
    """Start the webcam loop, detect faces, classify mask status, and display annotated frames."""

    trained_model = EfficientNet("B1", 4)
    trained_model.load_state_dict(torch.load(EFFICIENTNET_MODEL_PATH, map_location="cpu"))
    trained_model.to(device)
    trained_model.eval()

    base_options = python.BaseOptions(model_asset_path=BLAZEFACE_MODEL_PATH)
    options = vision.FaceDetectorOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO, min_detection_confidence=0.6)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    start_time = time.time()
    prev_frame_time = 0
    with vision.FaceDetector.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            fps = 1 / (current_time - prev_frame_time) if prev_frame_time != 0 else 0
            prev_frame_time = current_time
        
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb_frame.shape

            timestamp_ms = int((current_time - start_time) * 1000)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect_for_video(mp_image, timestamp_ms)
            
            if detection_result.detections:
                faces_tensor = []
                valid_boxes = []
                
                for detection in detection_result.detections:
                    bbox = detection.bounding_box
                    xmin = bbox.origin_x
                    ymin = bbox.origin_y
                    width = bbox.width
                    height = bbox.height
                    
                    pad_w = int(width * PAD_W_RATIO)
                    pad_h_top = int(height * PAD_H_TOP_RATIO)    
                    pad_h_bottom = int(height * PAD_H_BOTTOM_RATIO)  

                    x1 = max(0, int(xmin - pad_w))
                    y1 = max(0, int(ymin - pad_h_top))
                    x2 = min(w, int(xmin + width + pad_w))
                    y2 = min(h, int(ymin + height + pad_h_bottom))
                    
                    face_crop = rgb_frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue

                    pil_image = Image.fromarray(face_crop)
                    tensor_image = test_transform(pil_image)
                    
                    faces_tensor.append(tensor_image)
                    valid_boxes.append((x1, y1, x2, y2))
                
                if faces_tensor:
                    batch_tensors = torch.stack(faces_tensor).to(device)
                    
                    with torch.inference_mode():
                        outputs = trained_model(batch_tensors)
                        probs = torch.softmax(outputs, dim=1)
                        confidences, predicted_classes = torch.max(probs, dim=1)
                    
                    for i, box in enumerate(valid_boxes):
                        x1, y1, x2, y2 = box
                        class_idx = predicted_classes[i].item()
                        conf = confidences[i].item()
                        label_text = f"{LABELS[class_idx]} ({conf * 100:.1f}%)"
                        color = COLORS[class_idx]
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label_text, (x1, y1 - 15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            draw_hud(frame, device)

            cv2.imshow("Face Mask Real-time Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()