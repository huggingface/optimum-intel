#!/usr/bin/env python3
"""Box IoU / F1 accuracy-parity eval: OpenVINO vs PyTorch reference for LocateAnything-3B.

Treats the PyTorch (HF, trust_remote_code, bf16) slow-AR output as ground truth and
measures how closely each OpenVINO precision (FP16/INT8/INT4) on each device (CPU/GPU)
reproduces it: F1@0.5, F1@mean(IoU 0.5:0.05:0.95), mean box-corner L1, box-count match.

Methodology / fairness:
  * The exported vision IR bakes pos-emb + 2x2 patch-merge at the fixed 448x448 export
    resolution (256 image tokens). To compare apples-to-apples in one coordinate frame,
    BOTH the PyTorch reference and the OV pipeline preprocess every image at 448x448.
  * PyTorch reference: pure slow-AR greedy (use_cache=False), which keeps the vendored
    Qwen2 on its standard-causal branch (no magi / no PBD fast path) -- the in-scope path.
  * OV pipeline: vision IR -> scatter image features at <IMG_CONTEXT> -> text-embed IR ->
    stateful language IR with KV cache, greedy AR. (KV-cache greedy == slow-AR recompute
    for a causal model; validated by the Milestone-B 8/8 token match.)

Boxes parsed via regex <box><x1><y1><x2><y2></box>; coords are 0..1000 normalized.

Usage:
  python eval_box_parity.py --precisions fp16,int8,int4 --devices CPU,GPU.0
  python eval_box_parity.py --refresh-reference   # recompute the PyTorch GT cache
"""
import argparse
import json
import os
import re
import time

import numpy as np
from PIL import Image

RES = 448
EOS = 151645
IMG_TOKEN = 151665
MAX_NEW = 128
ART = "eval_artifacts"
BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")

# (image filename, query) pairs -- diverse detection + phrase-grounding queries.
SAMPLES = [
    ("000000039769.jpg", "cat"),
    ("000000039769.jpg", "remote control"),
    ("000000000139.jpg", "person"),
    ("000000000139.jpg", "tv"),
    ("000000000285.jpg", "bear"),
    ("000000000632.jpg", "bed"),
    ("000000000724.jpg", "stop sign"),
    ("000000001000.jpg", "person"),
    ("000000001268.jpg", "person"),
    ("000000001296.jpg", "car"),
    ("000000001503.jpg", "train"),
    ("000000002006.jpg", "the largest object in the image"),
    ("000000002149.jpg", "person"),
    ("000000002261.jpg", "bird"),
    ("000000002431.jpg", "person"),
]

IMG_DIR = "eval_images"


def build_prompt(query):
    return f"Locate all the instances that matches the following description: {query}."


def parse_boxes(text):
    """Return list of [x1,y1,x2,y2] in 0..1 normalized coords (from 0..1000 tokens)."""
    out = []
    for m in BOX_RE.finditer(text):
        x1, y1, x2, y2 = (int(v) / 1000.0 for v in m.groups())
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        out.append([x1, y1, x2, y2])
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def greedy_match(ov_boxes, gt_boxes):
    """Greedy IoU matching. Returns list of (ov_idx, gt_idx, iou) sorted desc."""
    pairs = []
    for i, ob in enumerate(ov_boxes):
        for j, gb in enumerate(gt_boxes):
            pairs.append((iou(ob, gb), i, j))
    pairs.sort(reverse=True)
    used_o, used_g, matches = set(), set(), []
    for v, i, j in pairs:
        if i in used_o or j in used_g:
            continue
        used_o.add(i)
        used_g.add(j)
        matches.append((i, j, v))
    return matches


def f1_at(matches, n_ov, n_gt, thr):
    tp = sum(1 for _, _, v in matches if v >= thr)
    fp = n_ov - tp
    fn = n_gt - tp
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 1.0  # both empty -> perfect agreement


def per_image_metrics(ov_boxes, gt_boxes):
    matches = greedy_match(ov_boxes, gt_boxes)
    n_ov, n_gt = len(ov_boxes), len(gt_boxes)
    f1_50 = f1_at(matches, n_ov, n_gt, 0.5)
    thrs = np.arange(0.5, 0.951, 0.05)
    f1_mean = float(np.mean([f1_at(matches, n_ov, n_gt, t) for t in thrs]))
    # corner L1 over matched pairs (in 0..1000 frame), only for IoU-matched pairs >0
    l1s = []
    for i, j, v in matches:
        if v > 0:
            a = np.array(ov_boxes[i]) * 1000.0
            b = np.array(gt_boxes[j]) * 1000.0
            l1s.append(float(np.abs(a - b).mean()))
    l1 = float(np.mean(l1s)) if l1s else None
    count_match = int(n_ov == n_gt)
    return {"f1_50": f1_50, "f1_mean": f1_mean, "corner_l1": l1,
            "count_match": count_match, "n_ov": n_ov, "n_gt": n_gt}


# --------------------------- PyTorch reference ---------------------------

def compute_reference(refresh=False):
    path = os.path.join(ART, "reference_boxes.json")
    if os.path.exists(path) and not refresh:
        return json.load(open(path))
    import torch
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    MODEL = "nvidia/LocateAnything-3B"
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.bfloat16, trust_remote_code=True).eval()
    model.tie_weights()
    idx = model.config.image_token_index
    ref = {}
    for fn, query in SAMPLES:
        key = f"{fn}::{query}"
        img = Image.open(os.path.join(IMG_DIR, fn)).convert("RGB").resize((RES, RES), Image.BICUBIC)
        messages = [{"role": "user", "content": [{"type": "image", "image": img},
                                                  {"type": "text", "text": build_prompt(query)}]}]
        text = proc.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = proc.process_vision_info(messages)
        inputs = proc(text=[text], images=images, videos=videos, return_tensors="pt")
        pv = inputs["pixel_values"].to(torch.bfloat16)
        ids = inputs["input_ids"]
        am = inputs["attention_mask"]
        gh = inputs["image_grid_hws"]
        if isinstance(gh, np.ndarray):
            gh = torch.from_numpy(gh)
        gh = gh.to(torch.int32)
        with torch.no_grad():
            vit = model.extract_feature(pv, gh)
            if isinstance(vit, (list, tuple)):
                vit = torch.cat(vit, dim=0)
            feats = model.mlp1(vit)
        gen = ids.clone()
        amask = am.clone()
        out = []
        t0 = time.time()
        for _ in range(MAX_NEW):
            with torch.no_grad():
                o = model.language_model(input_ids=gen,
                                         visual_features=feats.to(model.language_model.dtype),
                                         image_token_index=idx, attention_mask=amask, use_cache=False)
            nxt = int(o.logits[0, -1].argmax())
            out.append(nxt)
            if nxt == EOS:
                break
            gen = torch.cat([gen, torch.tensor([[nxt]], dtype=gen.dtype)], dim=1)
            amask = torch.cat([amask, torch.ones((1, 1), dtype=amask.dtype)], dim=1)
        decoded = tok.decode(out, skip_special_tokens=False)
        ref[key] = {"text": decoded, "boxes": parse_boxes(decoded),
                    "n_tokens": len(out), "seconds": round(time.time() - t0, 2),
                    "prompt_len": int(ids.shape[1])}
        print(f"[REF] {key}: {len(parse_boxes(decoded))} boxes, {len(out)} tok, {ref[key]['seconds']}s :: {decoded[:90]}")
    os.makedirs(ART, exist_ok=True)
    json.dump(ref, open(path, "w"), indent=2)
    return ref


# --------------------------- OV pipeline ---------------------------

class OVPipeline:
    def __init__(self, ir_dir, device):
        from optimum.intel.openvino import OVModelForVisualCausalLM
        from transformers import AutoProcessor, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(ir_dir, trust_remote_code=True)
        self.proc = AutoProcessor.from_pretrained(ir_dir, trust_remote_code=True)
        self.model = OVModelForVisualCausalLM.from_pretrained(ir_dir, trust_remote_code=True, device=device)
        self.vreq = self.model.vision_embeddings.request
        self.lreq = self.model.language_model.request

    def _inputs(self, fn, query):
        img = Image.open(os.path.join(IMG_DIR, fn)).convert("RGB").resize((RES, RES), Image.BICUBIC)
        messages = [{"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": build_prompt(query)}]}]
        text = self.proc.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.proc.process_vision_info(messages)
        inputs = self.proc(text=[text], images=images, videos=videos, return_tensors="pt")
        return (inputs["input_ids"].numpy(), inputs["attention_mask"].numpy(),
                inputs["pixel_values"].numpy().astype(np.float32),
                np.asarray(inputs["image_grid_hws"]).astype(np.int32))

    def generate(self, fn, query, max_new=MAX_NEW):
        ids, am, pv, gh = self._inputs(fn, query)
        vout = self.vreq({"pixel_values": pv, "image_grid_hws": gh})
        feats = list(vout.values())[0]
        # prefill embeddings with scattered vision features
        te = self.model.language_model.embed_tokens(ids.astype(np.int64))
        emb = np.array(te)[0].copy()
        sel = ids[0] == IMG_TOKEN
        emb[sel] = feats.astype(emb.dtype)
        S = ids.shape[1]
        mask = am.copy()
        self.lreq.reset_state()
        # prefill
        r = self.lreq.infer({"inputs_embeds": emb[None].astype(np.float32),
                             "attention_mask": mask.astype(np.int64),
                             "position_ids": np.arange(S, dtype=np.int64)[None],
                             "beam_idx": np.zeros(1, dtype=np.int32)})
        logits = list(r.values())[0][0]
        nxt = int(logits[-1].argmax())
        out = [nxt]
        pos = S
        t0 = time.time()
        ttft = None
        while nxt != EOS and len(out) < max_new:
            te1 = self.model.language_model.embed_tokens(np.array([[nxt]], dtype=np.int64))
            e1 = np.array(te1).astype(np.float32)
            mask = np.concatenate([mask, [[1]]], axis=1)
            r = self.lreq.infer({"inputs_embeds": e1,
                                 "attention_mask": mask.astype(np.int64),
                                 "position_ids": np.array([[pos]], dtype=np.int64),
                                 "beam_idx": np.zeros(1, dtype=np.int32)})
            if ttft is None:
                ttft = time.time() - t0
            logits = list(r.values())[0][0]
            nxt = int(logits[-1].argmax())
            out.append(nxt)
            pos += 1
        decoded = self.tok.decode(out, skip_special_tokens=False)
        return decoded, parse_boxes(decoded)


def run_matrix(precisions, devices, ref):
    rows = []
    for prec in precisions:
        ir_dir = f"ov_ir_{prec}"
        for dev in devices:
            print(f"\n=== OV {prec} on {dev} ===")
            pipe = OVPipeline(ir_dir, dev)
            per = []
            for fn, query in SAMPLES:
                key = f"{fn}::{query}"
                decoded, ov_boxes = pipe.generate(fn, query)
                m = per_image_metrics(ov_boxes, ref[key]["boxes"])
                per.append(m)
                print(f"  {key}: ov={m['n_ov']} gt={m['n_gt']} f1@.5={m['f1_50']:.3f} "
                      f"f1mean={m['f1_mean']:.3f} L1={m['corner_l1']}")
            f1_50 = float(np.mean([p["f1_50"] for p in per]))
            f1_mean = float(np.mean([p["f1_mean"] for p in per]))
            l1vals = [p["corner_l1"] for p in per if p["corner_l1"] is not None]
            corner_l1 = float(np.mean(l1vals)) if l1vals else None
            count_match = int(sum(p["count_match"] for p in per))
            row = {"precision": prec, "device": dev, "n_images": len(SAMPLES),
                   "f1_at_0.5": round(f1_50, 4), "f1_mean": round(f1_mean, 4),
                   "mean_corner_l1_px448": round(corner_l1, 2) if corner_l1 is not None else None,
                   "count_matched_images": count_match,
                   "per_image": per}
            rows.append(row)
            print(f"  -> {prec}/{dev}: F1@.5={f1_50:.4f} F1mean={f1_mean:.4f} "
                  f"L1={corner_l1} countmatch={count_match}/{len(SAMPLES)}")
            del pipe
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precisions", default="fp16,int8,int4")
    ap.add_argument("--devices", default="CPU")
    ap.add_argument("--refresh-reference", action="store_true")
    ap.add_argument("--out", default=os.path.join(ART, "box_parity_results.json"))
    args = ap.parse_args()
    os.makedirs(ART, exist_ok=True)

    ref = compute_reference(refresh=args.refresh_reference)
    precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    rows = run_matrix(precisions, devices, ref)

    summary = {"reference": "PyTorch nvidia/LocateAnything-3B bf16 slow-AR (GT)",
               "resolution": RES, "n_samples": len(SAMPLES),
               "table": [{k: v for k, v in r.items() if k != "per_image"} for r in rows],
               "rows": rows}
    json.dump(summary, open(args.out, "w"), indent=2)
    print("\n==== SUMMARY TABLE ====")
    print(f"{'prec':6} {'device':8} {'F1@0.5':8} {'F1@mean':8} {'L1(px448)':10} {'count':8}")
    for r in summary["table"]:
        print(f"{r['precision']:6} {r['device']:8} {r['f1_at_0.5']:<8} {r['f1_mean']:<8} "
              f"{str(r['mean_corner_l1_px448']):10} {r['count_matched_images']}/{r['n_images']}")
    print("saved", args.out)


if __name__ == "__main__":
    main()
