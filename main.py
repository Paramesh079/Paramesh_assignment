import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# ---------------------------- CONFIGURATION ----------------------------
csv_path = "C:/yt/file/Annotation-20250409T060352Z-001/Annotation/Annotation_For3Class.csv"
class_folder_map = {
    "ADVISORY SPEED MPH": "C:/yt/file/Dataset-20250409T060528Z-001/Dataset/ADVISORY SPEED MPH/ADVISORY SPEED MPH",
    "DIRECTIONAL ARROW AUXILIARY": "C:/yt/file/Dataset-20250409T060528Z-001/Dataset/DIRECTIONAL ARROW AUXILIARY/DIRECTIONAL ARROW AUXILIARY",
    "DO NOT ENTER": "C:/yt/file/Dataset-20250409T060528Z-001/Dataset/DO NOT ENTER/DO NOT ENTER"
}
filtered_images_dir = "C:/yt/filtered_images"
output_dir = "C:/yt/yolo_dataset"
images_output_dir = os.path.join(output_dir, 'images')
labels_output_dir = os.path.join(output_dir, 'labels')
data_yaml_path = os.path.join(output_dir, 'data.yaml')

class_names = list(class_folder_map.keys())
class_to_id = {name: idx for idx, name in enumerate(class_names)}

# ------------------Copy filtered images ------------------
print("Copying filtered images by class...")
os.makedirs(filtered_images_dir, exist_ok=True)

df = pd.read_csv(csv_path)
missing, copied = [], []

for _, row in df.iterrows():
    image_name = row['ImagePath']
    class_name = row['Class']
    src_path = os.path.join(class_folder_map[class_name], image_name)
    class_output_dir = os.path.join(filtered_images_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)
    dst_path = os.path.join(class_output_dir, image_name)

    if os.path.exists(src_path):
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
            copied.append(image_name)
    else:
        print(f"Image not found: {src_path}")
        missing.append(image_name)

print(f"\nDone. Copied {len(copied)} images into class folders.")
if missing:
    print(f"{len(missing)} images were not found.")

# ------------------ Create data.yaml ------------------
with open(data_yaml_path, 'w') as f:
    f.write(f"path: {output_dir}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("names: [" + ', '.join(f'"{name}"' for name in class_names) + "]\n")

print(f"data.yaml created at: {data_yaml_path}")

# ------------------ Split the dataset ------------------
df_image_class = df.groupby('ImagePath')['Class'].first().reset_index()

train_images, val_images = train_test_split(
    df_image_class['ImagePath'],
    test_size=0.2,
    random_state=42,
    stratify=df_image_class['Class']
)

train_images.to_csv('train_images.txt', index=False, header=False)
val_images.to_csv('val_images.txt', index=False, header=False)
print("Dataset split saved as train_images.txt and val_images.txt")

# ------------------ Convert to YOLO format ------------------
def convert_to_yolo_bbox(x0, x1, y0, y1, img_width, img_height):
    x_center = (x0 + x1) / 2.0 / img_width
    y_center = (y0 + y1) / 2.0 / img_height
    width = (x1 - x0) / img_width
    height = (y1 - y0) / img_height
    return x_center, y_center, width, height

# Make image and label directories
for split in ['train', 'val']:
    os.makedirs(os.path.join(images_output_dir, split), exist_ok=True)
    os.makedirs(os.path.join(labels_output_dir, split), exist_ok=True)

# Reload CSV
df = pd.read_csv(csv_path)
train_images = pd.read_csv('train_images.txt', header=None)[0].tolist()
val_images = pd.read_csv('val_images.txt', header=None)[0].tolist()

for split, image_list in zip(['train', 'val'], [train_images, val_images]):
    for image_name in tqdm(image_list, desc=f"Processing {split} set"):
        image_rows = df[df['ImagePath'] == image_name]
        class_name = image_rows.iloc[0]['Class']
        src_image_path = os.path.join(filtered_images_dir, class_name, image_name)

        if not os.path.exists(src_image_path):
            print(f"Missing image: {src_image_path}")
            continue

        dst_image_path = os.path.join(images_output_dir, split, image_name)
        shutil.copy2(src_image_path, dst_image_path)

        with Image.open(src_image_path) as img:
            img_width, img_height = img.size

        label_filename = os.path.splitext(image_name)[0] + '.txt'
        label_file_path = os.path.join(labels_output_dir, split, label_filename)

        with open(label_file_path, 'w') as f:
            for _, row in image_rows.iterrows():
                class_id = class_to_id[row['Class']]
                x_center, y_center, width, height = convert_to_yolo_bbox(
                    row['X0'], row['X1'], row['Y0'], row['Y1'], img_width, img_height
                )
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print("YOLO format conversion complete.")

# ------------------ Train YOLOv8 ------------------
print("Training started...")
model = YOLO("yolov8m.pt")
model.train(
    data=data_yaml_path,
    epochs=2,
    imgsz=928,
    batch=16,
    name="yolov8_model"
)
print("Training complete!")
