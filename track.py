from ultralytics import YOLO
from collections import Counter
from pathlib import Path

# Load model
model_path = "C:/yt/tech/runs/detect/train5/weights/best.pt"
model = YOLO(model_path)

# Video source
source = "C:/yt/tech/Object_Tracking-20250409T060354Z-001/Object_Tracking/Video/UnSeen_Dataset.mp4"

# Perform tracking
results = model.track(
    source=source,
    imgsz=928,
    conf=0.25,
    tracker="bytetrack.yaml",
    save=True,
    save_txt=True,
    persist=True
)

# Count detections per class
class_counts = Counter()
for result in results:
    if result.boxes is not None:
        for cls in result.boxes.cls:
            class_id = int(cls)
            class_name = model.names[class_id]
            class_counts[class_name] += 1

# Save counts to file
output_txt_path = Path(results[0].save_dir) / "class_counts.txt"
with open(output_txt_path, "w") as f:
    f.write("Detection count per class:\n")
    for class_name, count in class_counts.items():
        line = f"{class_name}: {count}"
        print(line)
        f.write(line + "\n")

print(f"\n Class counts saved to: {output_txt_path}")
