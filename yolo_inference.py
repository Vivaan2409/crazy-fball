from ultralytics import YOLO
import os

# Path to your trained model
model_path = r"D:\football_analysis\models\best.pt"  # your correct path

# Verify the file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Load the YOLOv8 model
model = YOLO(model_path)

# Path to your input video
video_path = r"D:\football_analysis\input_videos\08fd33_4.mp4"

# Output folder for inference
output_folder = r"D:\football_analysis\runs\detect\predict"
os.makedirs(output_folder, exist_ok=True)

# Run detection on the video
results = model.predict(
    source=video_path,
    save=True,        # saves output video to runs/detect/predict
    conf=0.3,         # confidence threshold
    show=False,       # True to display frames while processing
    stream=False      # True for frame-by-frame access
)

print("Inference complete. Check the output folder for the processed video.")

# Optional: Print detected boxes, classes, and confidence for analysis
for r in results:
    if r.boxes is None:
        continue
    for box in r.boxes:
        cls = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"Class: {cls}, Confidence: {conf:.2f}, Box: [{x1}, {y1}, {x2}, {y2}]")
