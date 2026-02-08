import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import os
import cv2
import numpy as np
from tqdm import tqdm
from filterpy.kalman import KalmanFilter

# --- كلاس بسيط للتتبع (Simplified SORT logic) ---
class SimpleTracker:
    def __init__(self):
        self.next_id = 1
        self.trackers = {} # {id: [x1, y1, x2, y2]}

    def update(self, detections):
        # detections: list of [x1, y1, x2, y2, score]
        updated_tracks = []
        for det in detections:
            # هنا نستخدم مبدأ الـ Centroid Tracking لتبسيط الكود على M2
            # في المشاريع الضخمة نستخدم Kalman Filter
            best_id = None
            min_dist = 50 
            
            cx, cy = (det[0]+det[2])/2, (det[1]+det[3])/2
            
            for tid, tbox in self.trackers.items():
                tcx, tcy = (tbox[0]+tbox[2])/2, (tbox[1]+tbox[3])/2
                dist = np.sqrt((cx-tcx)**2 + (cy-tcy)**2)
                if dist < min_dist:
                    min_dist = dist
                    best_id = tid
            
            if best_id is not None:
                self.trackers[best_id] = det[:4]
                updated_tracks.append(det[:4] + [best_id])
            else:
                self.trackers[self.next_id] = det[:4]
                updated_tracks.append(det[:4] + [self.next_id])
                self.next_id += 1
        return updated_tracks

# 1. الإعدادات
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model_path = "models/weights/final_model"
base_dir = 'data/raw/DETRAC-Images/DETRAC-Images' #
sequence_name = os.listdir(base_dir)[0] 
image_folder = os.path.join(base_dir, sequence_name)

# 2. تحميل الموديل والتركر
processor = DetrImageProcessor.from_pretrained(model_path)
model = DetrForObjectDetection.from_pretrained(model_path).to(device).eval()
tracker = SimpleTracker()

# 3. تجهيز الفيديو
output_path = f"tracking_{sequence_name}.mp4"
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])
frame = cv2.imread(os.path.join(image_folder, images[0]))
h, w, _ = frame.shape
video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))

print(f"🕵️ جاري تتبع المركبات في المجلد: {sequence_name}")

for img_name in tqdm(images):
    img_path = os.path.join(image_folder, img_name)
    pil_img = Image.open(img_path).convert("RGB")
    
    with torch.no_grad():
        inputs = processor(images=pil_img, return_tensors="pt").to(device)
        outputs = model(**inputs)
    
    results = processor.post_process_object_detection(outputs, target_sizes=torch.tensor([pil_img.size[::-1]]), threshold=0.6)[0]
    
    # تحويل النتائج لتناسب التركر
    dets = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        dets.append(box.tolist() + [score.item()])
    
    # تحديث التركر
    tracks = tracker.update(dets)
    
    # الرسم
    draw_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    for t in tracks:
        x1, y1, x2, y2, obj_id = map(int, t)
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(draw_img, f"ID: {obj_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    video.write(draw_img)

video.release()
print(f"✅ انتهى! الفيديو المسجل مع الـ IDs موجود هنا: {output_path}")
os.system(f"open {output_path}")