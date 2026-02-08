import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import os
import random

# 1. إعداد الجهاز
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"🍏 Using device: {device}")

# 2. مسار الموديل النهائي (الذي انتهى تدريبه للتو)
model_path = "models/weights/final_model"
image_dir = 'data/raw/DETRAC-Images/DETRAC-Images'

# 3. تحميل الموديل
print(f"🔄 جاري تحميل الموديل المدرب من: {model_path}")
try:
    processor = DetrImageProcessor.from_pretrained(model_path)
    model = DetrForObjectDetection.from_pretrained(model_path)
    model.to(device)
    model.eval()
except Exception as e:
    print(f"❌ خطأ: لم يتم العثور على الموديل في {model_path}")
    print("تأكد أن التدريب انتهى وأن المجلد موجود.")
    exit()

# 4. اختيار صورة عشوائية للاختبار
def get_random_image(root_dir):
    all_images = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(root, file))
    return random.choice(all_images) if all_images else None

image_path = get_random_image(image_dir)
if not image_path:
    print("❌ لا توجد صور في المجلد!")
    exit()

print(f"📸 جاري اختبار الصورة: {image_path}")
image = Image.open(image_path).convert("RGB")

# 5. الكشف (Inference)
with torch.no_grad():
    # نرسل الصورة للموديل
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)

# 6. معالجة النتائج
target_sizes = torch.tensor([image.size[::-1]]).to(device)
# نرفع العتبة (Threshold) إلى 0.5 لنرى فقط التوقعات القوية
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]

# 7. الرسم على الصورة
draw = ImageDraw.Draw(image)
try:
    font = ImageFont.truetype("arial.ttf", 24)
except:
    font = ImageFont.load_default()

print(f"🎯 تم اكتشاف {len(results['scores'])} مركبة:")

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    
    # تحويل الرقم إلى اسم (Car, Bus...)
    label_name = model.config.id2label[label.item()]
    confidence = round(score.item(), 2)
    
    print(f" - {label_name}: {confidence}% at {box}")
    
    # تغيير اللون حسب النوع
    color = "red"
    if label_name == "Bus": color = "blue"
    if label_name == "Van": color = "green"

    # رسم المربع
    draw.rectangle(box, outline=color, width=4)
    # رسم الخلفية للنص ليكون واضحاً
    text_origin = (box[0], box[1] - 25)
    draw.rectangle([text_origin, (text_origin[0] + 100, text_origin[1] + 25)], fill=color)
    draw.text(text_origin, f"{label_name} {confidence}", fill="white", font=font)

# 8. حفظ وعرض النتيجة
output_path = "final_result.jpg"
image.save(output_path)
print(f"✅ تم الحفظ في: {output_path}")
os.system(f"open {output_path}")