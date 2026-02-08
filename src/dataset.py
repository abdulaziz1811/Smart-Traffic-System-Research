"""
Dataset: Preparation · Loading · Augmentation
===============================================
- convert_detrac()   → XML → COCO JSON (with train/val/test split)
- DetrDataset        → PyTorch Dataset
- get_dataloaders()  → ready-to-use DataLoaders

⚠️ Backward compatible: works with existing data/annotations/train.json
   If val.json/test.json don't exist, only train_loader is returned.
"""

import os, json, random, logging
import xml.etree.ElementTree as ET
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CocoDetection
from transformers import DetrImageProcessor
from PIL import Image
from tqdm import tqdm

log = logging.getLogger("traffic")


# ════════════════════════════════════════════════════════════
#  PART 1 — Data Preparation
# ════════════════════════════════════════════════════════════

def _img_size(path, default):
    if os.path.isfile(path):
        try:
            with Image.open(path) as im: return im.size
        except: pass
    return default


def _parse_xml(xml_path, img_dir, cat_map, default_sz, verify=False):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    seq = root.get("name")
    imgs, anns, stats = [], [], defaultdict(int)

    for frame in root.findall("frame"):
        num = int(frame.get("num"))
        rel = f"{seq}/img{num:05d}.jpg"
        full = os.path.join(img_dir, rel)
        if verify and not os.path.isfile(full): continue

        w, h = _img_size(full, default_sz)
        imgs.append({"file_name": rel, "width": w, "height": h})

        frame_anns = []
        tl = frame.find("target_list")
        if tl is not None:
            for t in tl.findall("target"):
                bx = t.find("box"); attr = t.find("attribute")
                bbox = [float(bx.get("left")), float(bx.get("top")),
                        float(bx.get("width")), float(bx.get("height"))]
                vtype = attr.get("vehicle_type", "others").lower()
                cid = cat_map.get(vtype, cat_map.get("others", 4))
                stats[vtype] += 1
                frame_anns.append({"category_id": cid, "bbox": bbox,
                                   "area": bbox[2]*bbox[3], "iscrowd": 0})
        anns.append(frame_anns)
    return imgs, anns, dict(stats)


def _to_coco(imgs, ann_groups, cat_map):
    coco = {"images": [], "annotations": [],
            "categories": [{"id": v, "name": k} for k, v in cat_map.items()]}
    iid = aid = 1
    for entry, ag in zip(imgs, ann_groups):
        coco["images"].append({"id": iid, **entry})
        for a in ag:
            coco["annotations"].append({"id": aid, "image_id": iid, **a}); aid += 1
        iid += 1
    return coco


def convert_detrac(cfg: dict, verify_images: bool = False):
    """Parse XMLs → sequence-level split → save COCO JSONs."""
    p = cfg["paths"]; ds = cfg["dataset"]; seed = cfg["training"]["seed"]
    cat_map = ds["categories"]; default_sz = tuple(ds["default_image_size"])

    xml_files = sorted(f for f in os.listdir(p["xml_dir"]) if f.endswith(".xml"))
    if not xml_files: log.error(f"No XMLs in {p['xml_dir']}"); return

    log.info(f"Parsing {len(xml_files)} XML files …")

    seq_data = {}; totals = defaultdict(int)
    for xf in tqdm(xml_files, desc="Parsing"):
        try:
            ims, ans, st = _parse_xml(os.path.join(p["xml_dir"], xf),
                                       p["images_dir"], cat_map, default_sz, verify_images)
            seq_data[xf.replace(".xml","")] = (ims, ans)
            for k,v in st.items(): totals[k] += v
        except Exception as e:
            log.warning(f"Skip {xf}: {e}")

    # Class distribution
    total = sum(totals.values())
    log.info("── Class Distribution ──")
    for cls, cnt in sorted(totals.items(), key=lambda x: -x[1]):
        log.info(f"  {cls:>10s}: {cnt:>8,d}  ({cnt/total*100:.1f}%)")

    # Sequence-level split (prevents data leakage)
    seqs = sorted(seq_data.keys()); rng = random.Random(seed); rng.shuffle(seqs)
    r = ds["split_ratios"]; n = len(seqs)
    nt = int(n*r["train"]); nv = int(n*r["val"])
    splits = {"train": seqs[:nt], "val": seqs[nt:nt+nv], "test": seqs[nt+nv:]}

    out = p["annotations_dir"]; os.makedirs(out, exist_ok=True)
    for name, sl in splits.items():
        ai, aa = [], []
        for s in sl:
            if s in seq_data: i,a = seq_data[s]; ai.extend(i); aa.extend(a)
        coco = _to_coco(ai, aa, cat_map)
        fp = os.path.join(out, f"{name}.json")
        with open(fp, "w") as f: json.dump(coco, f)
        log.info(f"  {name}: {len(coco['images']):,} imgs, {len(coco['annotations']):,} anns → {fp}")

    meta = {"seed": seed, "ratios": r, "sequences": {k: sorted(v) for k,v in splits.items()}}
    with open(os.path.join(out, "split_info.json"), "w") as f: json.dump(meta, f, indent=2)
    log.info("Split info → split_info.json")


# ════════════════════════════════════════════════════════════
#  PART 2 — Augmentation
# ════════════════════════════════════════════════════════════

class Augmentation:
    def __init__(self, cfg, training=True):
        aug = cfg.get("augmentation", {}); self.training = training
        self.flip = aug.get("horizontal_flip", False) and training
        self.flip_p = aug.get("flip_prob", 0.5)
        cj = aug.get("color_jitter", {})
        self.jitter = (T.ColorJitter(cj.get("brightness",0), cj.get("contrast",0),
                                     cj.get("saturation",0), cj.get("hue",0))
                       if cj.get("enabled") and training else None)

    def __call__(self, image, target):
        if self.jitter: image = self.jitter(image)
        if self.flip and torch.rand(1).item() < self.flip_p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            w = image.width
            for a in target.get("annotations", []):
                b = a["bbox"]; a["bbox"] = [w-b[0]-b[2], b[1], b[2], b[3]]
        return image, target


# ════════════════════════════════════════════════════════════
#  PART 3 — Dataset & DataLoaders
# ════════════════════════════════════════════════════════════

class DetrDataset(CocoDetection):
    def __init__(self, img_dir, ann_file, processor, aug=None):
        super().__init__(img_dir, ann_file)
        self.processor = processor; self.aug = aug

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        target = {"image_id": self.ids[idx], "annotations": target}
        if self.aug: img, target = self.aug(img, target)
        enc = self.processor(images=img, annotations=target, return_tensors="pt")
        return {"pixel_values": enc["pixel_values"].squeeze(0), "labels": enc["labels"][0]}


def _collate(batch, processor):
    pv = [b["pixel_values"] for b in batch]
    enc = processor.pad(pv, return_tensors="pt")
    return {"pixel_values": enc["pixel_values"], "pixel_mask": enc["pixel_mask"],
            "labels": [b["labels"] for b in batch]}


def get_processor(cfg):
    m = cfg["model"]
    return DetrImageProcessor.from_pretrained(
        m["backbone"], do_resize=True,
        size={"shortest_edge": m["image_size"]["shortest_edge"],
              "longest_edge": m["image_size"]["longest_edge"]},
        image_mean=m["image_mean"], image_std=m["image_std"])


def get_dataloaders(cfg, processor=None):
    """Returns (train_loader, val_loader|None, test_loader|None)."""
    if processor is None: processor = get_processor(cfg)
    dl_cfg = cfg["training"]["dataloader"]; bs = cfg["training"]["batch_size"]
    collate = partial(_collate, processor=processor)
    loaders = []
    for split, is_train in [("train",True), ("val",False), ("test",False)]:
        ann = os.path.join(cfg["paths"]["annotations_dir"], f"{split}.json")
        if not os.path.isfile(ann): loaders.append(None); continue
        ds = DetrDataset(cfg["paths"]["images_dir"], ann, processor, Augmentation(cfg, is_train))
        log.info(f"{split.capitalize()}: {len(ds):,} images")
        loaders.append(DataLoader(
            ds, batch_size=bs, shuffle=is_train,
            num_workers=dl_cfg.get("num_workers",0), collate_fn=collate,
            pin_memory=dl_cfg.get("pin_memory",False) and torch.cuda.is_available(),
            drop_last=dl_cfg.get("drop_last",False) and is_train))
    return tuple(loaders)
