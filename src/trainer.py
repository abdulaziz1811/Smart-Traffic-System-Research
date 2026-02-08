"""
Training & Evaluation
======================
Trainer.fit() → train/val loop → mAP → early stopping → checkpoints

Saves to: models/weights/{best_model, final_model, checkpoints/}
"""

import os, json, time, shutil, logging
from typing import Dict, List, Optional, Tuple

import torch, torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, StepLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

log = logging.getLogger("traffic")


# ── mAP Evaluator ───────────────────────────────────────

@torch.no_grad()
def evaluate_map(model, processor, loader, device):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        log.warning("pycocotools missing"); return {"mAP@50:95":0., "mAP@50":0., "mAP@75":0.}

    model.eval(); preds, gts = [], []; iid = 0
    for batch in tqdm(loader, desc="Eval", leave=False):
        pv = batch["pixel_values"].to(device)
        pm = batch["pixel_mask"].to(device)
        out = model(pixel_values=pv, pixel_mask=pm)
        for i, lab in enumerate(batch["labels"]):
            iid += 1; orig = lab["orig_size"]
            ts = torch.tensor([orig.tolist()]).to(device)
            res = processor.post_process_object_detection(
                {"logits": out.logits[i:i+1], "pred_boxes": out.pred_boxes[i:i+1]},
                target_sizes=ts, threshold=0.01)[0]
            for sc, lb, bx in zip(res["scores"], res["labels"], res["boxes"]):
                x1,y1,x2,y2 = bx.tolist()
                preds.append({"image_id":iid, "category_id":lb.item(),
                              "bbox":[x1,y1,x2-x1,y2-y1], "score":sc.item()})
            h, w = orig.tolist()
            for j in range(len(lab["class_labels"])):
                cx,cy,bw,bh = lab["boxes"][j].tolist()
                gts.append({"id":len(gts)+1, "image_id":iid,
                            "category_id":lab["class_labels"][j].item(),
                            "bbox":[(cx-bw/2)*w,(cy-bh/2)*h,bw*w,bh*h],
                            "area":bw*w*bh*h, "iscrowd":0})

    if not preds or not gts: return {"mAP@50:95":0., "mAP@50":0., "mAP@75":0.}
    gt_o = COCO(); gt_o.dataset = {
        "images":[{"id":i} for i in {g["image_id"] for g in gts}],
        "annotations":gts, "categories":[{"id":c} for c in {g["category_id"] for g in gts}]}
    gt_o.createIndex(); dt_o = gt_o.loadRes(preds)
    ev = COCOeval(gt_o, dt_o, "bbox"); ev.evaluate(); ev.accumulate(); ev.summarize()
    return {"mAP@50:95":ev.stats[0], "mAP@50":ev.stats[1], "mAP@75":ev.stats[2],
            "mAP_small":ev.stats[3], "mAP_medium":ev.stats[4], "mAP_large":ev.stats[5]}


# ── Trainer ──────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=5, delta=0.001):
        self.patience, self.delta = patience, delta
        self.best, self.wait = None, 0
    def __call__(self, score):
        if self.best is None or score > self.best+self.delta:
            self.best, self.wait = score, 0; return False
        self.wait += 1; return self.wait >= self.patience


class Trainer:
    def __init__(self, model, processor, cfg, device):
        self.model, self.processor, self.cfg, self.device = model, processor, cfg, device
        tc = cfg["training"]
        self.epochs, self.grad_accum = tc["epochs"], tc["gradient_accumulation"]
        self.max_norm = tc.get("max_grad_norm", 0.1)

        wd = cfg["paths"]["weights_dir"]
        self.ckpt_dir = os.path.join(wd,"checkpoints")
        self.best_dir = os.path.join(wd,"best_model")
        self.final_dir = os.path.join(wd,"final_model")
        for d in (self.ckpt_dir, self.best_dir, self.final_dir): os.makedirs(d, exist_ok=True)

        self.optim = AdamW(model.parameters(), lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
        es = tc.get("early_stopping",{})
        self.early = EarlyStopping(es.get("patience",5), es.get("min_delta",0.001)) if es.get("enabled") else None

        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(os.path.join(cfg["paths"]["log_dir"],"tb"))
        except: self.tb = None

        self.step, self.best_map = 0, 0.
        self.keep_n = tc.get("checkpointing",{}).get("keep_last_n",3)

    def _sched(self, spe):
        tc = self.cfg["training"]; total = spe*self.epochs//self.grad_accum
        warmup = int(total*tc.get("warmup_ratio",0.05))
        warm = LinearLR(self.optim, start_factor=0.1, total_iters=warmup)
        s = tc.get("scheduler","cosine")
        if s=="cosine": main = CosineAnnealingLR(self.optim, T_max=total-warmup)
        elif s=="step": main = StepLR(self.optim, step_size=max((total-warmup)//3,1), gamma=0.1)
        else: main = LinearLR(self.optim, start_factor=1., end_factor=0., total_iters=total-warmup)
        self.scheduler = SequentialLR(self.optim, [warm,main], milestones=[warmup])

    def _train_ep(self, ld, ep):
        self.model.train(); tot,n = 0.,0; self.optim.zero_grad()
        bar = tqdm(ld, desc=f"Epoch {ep+1}/{self.epochs}")
        for i, batch in enumerate(bar):
            pv = batch["pixel_values"].to(self.device)
            pm = batch["pixel_mask"].to(self.device)
            labels = [{k:v.to(self.device) for k,v in t.items()} for t in batch["labels"]]
            out = self.model(pixel_values=pv, pixel_mask=pm, labels=labels)
            (out.loss/self.grad_accum).backward()
            if (i+1)%self.grad_accum==0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.optim.step(); self.scheduler.step(); self.optim.zero_grad(); self.step+=1
                if self.tb and self.step%50==0:
                    self.tb.add_scalar("train/loss",out.loss.item(),self.step)
            tot+=out.loss.item(); n+=1; bar.set_postfix(loss=f"{out.loss.item():.4f}")
        return tot/max(n,1)

    @torch.no_grad()
    def _val(self, ld):
        self.model.eval(); tot,n = 0.,0
        for batch in tqdm(ld, desc="Val", leave=False):
            pv = batch["pixel_values"].to(self.device)
            pm = batch["pixel_mask"].to(self.device)
            labels = [{k:v.to(self.device) for k,v in t.items()} for t in batch["labels"]]
            tot+=self.model(pixel_values=pv, pixel_mask=pm, labels=labels).loss.item(); n+=1
        avg = tot/max(n,1)
        metrics = evaluate_map(self.model, self.processor, ld, self.device)
        metrics["val_loss"] = avg
        return avg, metrics

    def _save(self, path, ep, m):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path); self.processor.save_pretrained(path)
        with open(os.path.join(path,"state.json"),"w") as f:
            json.dump({"epoch":ep,"step":self.step,"best_map":self.best_map,"metrics":m},f,indent=2)

    def _prune(self):
        dirs = sorted(d for d in os.listdir(self.ckpt_dir) if d.startswith("epoch-"))
        while len(dirs)>self.keep_n:
            shutil.rmtree(os.path.join(self.ckpt_dir,dirs.pop(0)), ignore_errors=True)

    def fit(self, train_ld, val_ld=None):
        self._sched(len(train_ld))
        log.info("═"*55)
        log.info(f"Training: {self.epochs} epochs, batch×accum={self.cfg['training']['batch_size']}×{self.grad_accum}")
        log.info("═"*55)

        hist = {"train_loss":[],"val_loss":[],"mAP@50":[],"mAP@75":[],"mAP@50:95":[]}
        for ep in range(self.epochs):
            t0 = time.time()
            tl = self._train_ep(train_ld, ep); hist["train_loss"].append(tl)
            log.info(f"Epoch {ep+1} — train_loss={tl:.4f} ({time.time()-t0:.0f}s)")

            m = {}
            if val_ld:
                vl, m = self._val(val_ld); hist["val_loss"].append(vl)
                for k in ("mAP@50","mAP@75","mAP@50:95"): hist[k].append(m.get(k,0.))
                log.info(f"  val={vl:.4f} mAP@50={m.get('mAP@50',0):.4f} mAP@75={m.get('mAP@75',0):.4f}")
                if self.tb:
                    self.tb.add_scalar("val/loss",vl,ep)
                    for mk,mv in m.items():
                        if isinstance(mv,(int,float)): self.tb.add_scalar(f"val/{mk}",mv,ep)
                cur = m.get("mAP@50",0.)
                if cur>self.best_map:
                    self.best_map=cur; self._save(self.best_dir,ep,m)
                    log.info(f"  ★ New best mAP@50={self.best_map:.4f}")
                if self.early and self.early(cur): log.info(f"Early stop ep {ep+1}"); break

            self._save(os.path.join(self.ckpt_dir,f"epoch-{ep+1:03d}"),ep,m); self._prune()

        self._save(self.final_dir,self.epochs,m)
        with open(os.path.join(self.cfg["paths"]["weights_dir"],"history.json"),"w") as f:
            json.dump(hist,f,indent=2)
        if self.tb: self.tb.close()
        log.info("Training complete ✓"); return hist
