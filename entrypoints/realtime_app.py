
import os
import sys
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

def ensure_model_exists(path, url):
    if not os.path.exists(path):
        print(f"Model file '{path}' not found. Downloading from network...")
        try:
            urllib.request.urlretrieve(url, path)
            print("Download completed successfully!")
        except Exception as e:
            print(f"Download failed. Error: {e}")
            raise SystemExit(1)

ensure_model_exists(BLAZEFACE_MODEL_PATH, BLAZEFACE_MODEL_URL)

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

trained_model = EfficientNet("B1", 4)
trained_model.load_state_dict(torch.load(EFFICIENTNET_MODEL_PATH, map_location="cpu"))
trained_model.to(device)
trained_model.eval()

base_options = python.BaseOptions(model_asset_path=BLAZEFACE_MODEL_PATH)
options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.6)
detector = vision.FaceDetector.create_from_options(options)

LABELS = ["Mask on chin", "Mask not covering nose", "Mask properly worn", "No mask"]
COLORS = [
    (0, 165, 255),  # 橘色: Mask on chin
    (0, 165, 255),  # 橘色: Mask not covering nose
    (0, 255, 0),    # 綠色: Mask properly worn
    (0, 0, 255)     # 紅色: No mask
]

def main():
    """Start the webcam loop, detect faces, classify mask status, and display annotated frames."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb_frame.shape

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        
        if detection_result.detections:
            faces_tensor = []
            valid_boxes = []
            
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                xmin = bbox.origin_x
                ymin = bbox.origin_y
                width = bbox.width
                height = bbox.height
                
                pad_w, pad_h = int(width * 0.2), int(height * 0.2)
                x1 = max(0, xmin - pad_w)
                y1 = max(0, ymin - pad_h)
                x2 = min(w, xmin + width + pad_w)
                y2 = min(h, ymin + height + pad_h)
                
                face_crop = rgb_frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                pil_image = Image.fromarray(face_crop)
                tensor_image = test_transform(pil_image)
                tensor_image = tensor_image.reshape(-1, 3, 240, 240)
                
                faces_tensor.append(tensor_image)
                valid_boxes.append((x1, y1, x2, y2))
            
            if faces_tensor:
                batch_tensors = torch.cat(faces_tensor).to(device)
                
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

        cv2.imshow("Face Mask Real-time Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()