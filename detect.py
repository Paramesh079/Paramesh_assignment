##for video

from ultralytics import YOLO


model_path = "C:/yt/tech/runs/detect/train5/weights/best.pt"  
model = YOLO(model_path)

#for video
#source = 'Object_Tracking-20250409T060354Z-001/Object_Tracking/Video/UnSeen_Dataset.mp4' 

#for folder of images
source = 'C:/yt/tech/Object_Tracking-20250409T060354Z-001/Object_Tracking/Images'

results = model.predict(
    source=source,
    imgsz=928,
    conf=0.25,
    save=True,         
    save_txt=True        
)
print("Prediction complete. Results saved in:", results[0].save_dir)