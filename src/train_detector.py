import torch
from transformers import DetrImageProcessor, DetrForObjectDetection, Trainer, TrainingArguments
from torchvision.datasets import CocoDetection
import os

# 1. إعداد الجهاز
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"🍏 يتم استخدام المعالج: {device}")

# 2. المسارات
img_folder = 'data/raw/DETRAC-Images/DETRAC-Images'
ann_file = 'data/annotations/train.json'

# 3. تعريف التصنيفات
id2label = {1: "Car", 2: "Bus", 3: "Van", 4: "Others"}
label2id = {"Car": 1, "Bus": 2, "Van": 3, "Others": 4}

# 4. تحميل الموديل والمعالج (نسخة Lite)
print("🔄 جاري إعداد الموديل بنسخة 'Lite' لتناسب M2...")

processor = DetrImageProcessor.from_pretrained(
    "facebook/detr-resnet-50",
    do_resize=True,
    size={"shortest_edge": 480, "longest_edge": 640}, # تصغير الصور هو الحل السحري
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225]
)

model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
model.to(device)

# 5. تجهيز البيانات
class DetrDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, processor):
        super(DetrDataset, self).__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        img, target = super(DetrDataset, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]
        return {"pixel_values": pixel_values, "labels": target}

train_dataset = DetrDataset(img_folder, ann_file, processor)

# 6. إعدادات التدريب المخصصة للماك (النسخة المستقرة)
training_args = TrainingArguments(
    output_dir="models/weights/detr_finetuned",
    per_device_train_batch_size=1,   # دفعة صغيرة
    gradient_accumulation_steps=8,   # تجميع التحديثات
    num_train_epochs=1,
    save_steps=500,
    logging_steps=50,
    learning_rate=1e-4,
    weight_decay=1e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none",
    dataloader_pin_memory=False,
    gradient_checkpointing=False,    # [تم التعطيل] لتفادي الخطأ
    dataloader_num_workers=0
)

def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    return {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": labels
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
)

print("🚀 انطلاق التدريب المخفف (المحاولة الثالثة ثابتة!)...")
trainer.train()

# حفظ النموذج النهائي
model.save_pretrained("models/weights/final_model")
processor.save_pretrained("models/weights/final_model")
print("✅ تم الانتهاء!")