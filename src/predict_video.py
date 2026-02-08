import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import os
import cv2
import numpy as np
from tqdm import tqdm

# 1. إعداد الجهاز والموديل
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"🍏 Using device: {device}")

model_path = "models/weights/final_model"
# سنختار مجلداً واحداً للتجربة (يمكنك تغيير الرقم لمجلد آخر مثل MVI_40131)
# هذا الكود يبحث عن أول مجلد يجده تلقائياً
base_dir = 'data/raw/DETRAC-Images/DETRAC-Images'
available_dirs = [d for d in os.listdir(base_dir) if d.startswith("MVI")]
if not available_dirs:
    print("❌ لم يتم العثور على مجلدات MVI في المسار!")
    exit()

# نختار مجلد عشوائي أو محدد (مثلاً MVI_40131 المشهور)
sequence_name = available_dirs[0] 
image_folder = os.path.join(base_dir, sequence_name)
print(f"🎬 جاري إنشاء فيديو للمجلد: {sequence_name}")

# 2. تحميل الموديل
try:
    processor = DetrImageProcessor.from_pretrained(model_path)
    model = DetrForObjectDetection.from_pretrained(model_path)
    model.to(device)
    model.eval()
except:
    print("❌ تأكد من مسار الموديل!")
    exit()

# 3. إعداد الفيديو
output_video_path = f"traffic_analysis_{sequence_name}.mp4"
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])

if not images:
    print("❌ المجلد فارغ!")
    exit()

# قراءة الصورة الأولى لمعرفة الأبعاد
first_frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = first_frame.shape
fps = 25 # سرعة الفيديو

# استخدام كودك لضغط الفيديو (mp4v يعمل جيداً على الماك)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# 4. معالجة الصور واحدة تلو الأخرى
print(f"🚀 جاري معالجة {len(images)} إطار... (قد يستغرق بضع دقائق)")

for img_name in tqdm(images):
    img_path = os.path.join(image_folder, img_name)
    
    # فتح الصورة بـ PIL للموديل
    pil_image = Image.open(img_path).convert("RGB")
    
    # التنبؤ
    with torch.no_grad():
        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        outputs = model(**inputs)
        
    target_sizes = torch.tensor([pil_image.size[::-1]]).to(device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

    # الرسم باستخدام OpenCV (أسرع للفيديو)
    # نحول من PIL إلى OpenCV
    opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [int(i) for i in box.tolist()]
        label_name = model.config.id2label[label.item()]
        conf = round(score.item(), 2)

        # ألوان مختلفة
        color = (0, 0, 255) # أحمر للسيارات (BGR)
        if label_name == "Bus": color = (255, 0, 0) # أزرق
        if label_name == "Van": color = (0, 255, 0) # أخضر

        # رسم المربع
        cv2.rectangle(opencv_image, (box[0], box[1]), (box[2], box[3]), color, 2)
        
        # كتابة النص
        label_text = f"{label_name} {conf}"
        cv2.putText(opencv_image, label_text, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # إضافة الإطار للفيديو
    video.write(opencv_image)

# 5. إنهاء وحفظ
video.release()
cv2.destroyAllWindows()
print(f"✅ تم حفظ الفيديو بنجاح: {output_video_path}")
os.system(f"open {output_video_path}")