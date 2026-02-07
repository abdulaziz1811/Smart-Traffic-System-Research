import os
import xml.etree.ElementTree as ET
import json
from tqdm import tqdm

def convert_detrac_to_coco(xml_dir, img_base_dir, save_path):
    if not os.path.exists(xml_dir):
        print(f"❌ المسار غير موجود: {xml_dir}")
        return

    coco = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "car"},
            {"id": 2, "name": "bus"},
            {"id": 3, "name": "van"},
            {"id": 4, "name": "others"}
        ]
    }
    
    cat_map = {"car": 1, "bus": 2, "van": 3, "others": 4}
    ann_id = 1
    img_id = 1

    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    print(f"🔍 تم العثور على {len(xml_files)} ملف XML. بدء التحويل...")

    for xml_file in tqdm(xml_files, desc="Converting"):
        try:
            tree = ET.parse(os.path.join(xml_dir, xml_file))
            root = tree.getroot()
            seq_name = root.get('name')
            
            for frame in root.findall('frame'):
                frame_num = int(frame.get('num'))
                # المسار النسبي للصورة داخل مجلد الصور
                img_path = f"{seq_name}/img{frame_num:05d}.jpg"
                
                coco["images"].append({
                    "id": img_id,
                    "file_name": img_path,
                    "width": 960,
                    "height": 540
                })
                
                target_list = frame.find('target_list')
                if target_list is not None:
                    for target in target_list.findall('target'):
                        box = target.find('box')
                        attr = target.find('attribute')
                        
                        bbox = [
                            float(box.get('left')),
                            float(box.get('top')),
                            float(box.get('width')),
                            float(box.get('height'))
                        ]
                        
                        v_type = attr.get('vehicle_type')
                        category_id = cat_map.get(v_type, 4)
                        
                        coco["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": category_id,
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0
                        })
                        ann_id += 1
                img_id += 1
        except Exception as e:
            print(f"⚠️ خطأ في {xml_file}: {e}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(coco, f)
    print(f"\n✅ تم بنجاح! السجلات: {len(coco['images'])} صورة، {len(coco['annotations'])} تصنيف.")

if __name__ == "__main__":
    # تم تحديث المسارات بناءً على مخرجات ls -R اللي أرسلتها
    convert_detrac_to_coco(
        xml_dir='data/annotations/DETRAC-Train-Annotations-XML/DETRAC-Train-Annotations-XML', 
        img_base_dir='data/raw', 
        save_path='data/annotations/train.json'
    )