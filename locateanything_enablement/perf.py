#!/usr/bin/env python3
"""Quick informational perf for the LocateAnything-3B slow-AR LLM (OV).

Measures mean TTFT (prefill -> first generated token) and per-token ITL
(inter-token latency, KV-cache decode steps) for the language IR, on a fixed
image+query sample, across precision x device. Vision/embeds excluded from the
LLM timing (reported separately as prep time).

Usage: python perf.py --precisions fp16,int4 --devices CPU,GPU.0 --runs 3
"""
import argparse
import json
import os
import time

import numpy as np
from PIL import Image

RES = 448
EOS = 151645
IMG_TOKEN = 151665
IMG_DIR = "eval_images"
SAMPLE = ("000000001268.jpg", "person")  # multi-box -> a few dozen decode steps
ART = "eval_artifacts"


def build_prompt(q):
    return f"Locate all the instances that matches the following description: {q}."


class Pipe:
    def __init__(self, ir_dir, device):
        from optimum.intel.openvino import OVModelForVisualCausalLM
        from transformers import AutoProcessor, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(ir_dir, trust_remote_code=True)
        self.proc = AutoProcessor.from_pretrained(ir_dir, trust_remote_code=True)
        self.model = OVModelForVisualCausalLM.from_pretrained(ir_dir, trust_remote_code=True, device=device)
        self.vreq = self.model.vision_embeddings.request
        self.lreq = self.model.language_model.request

    def prep(self, fn, q):
        img = Image.open(os.path.join(IMG_DIR, fn)).convert("RGB").resize((RES, RES), Image.BICUBIC)
        messages = [{"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": build_prompt(q)}]}]
        text = self.proc.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.proc.process_vision_info(messages)
        inputs = self.proc(text=[text], images=images, videos=videos, return_tensors="pt")
        ids = inputs["input_ids"].numpy()
        am = inputs["attention_mask"].numpy()
        pv = inputs["pixel_values"].numpy().astype(np.float32)
        gh = np.asarray(inputs["image_grid_hws"]).astype(np.int32)
        vout = self.vreq({"pixel_values": pv, "image_grid_hws": gh})
        feats = list(vout.values())[0]
        te = self.model.language_model.embed_tokens(ids.astype(np.int64))
        emb = np.array(te)[0].copy()
        emb[ids[0] == IMG_TOKEN] = feats.astype(emb.dtype)
        return ids, am, emb

    def run_once(self, ids, am, emb, max_new=128):
        S = ids.shape[1]
        mask = am.copy()
        self.lreq.reset_state()
        t0 = time.time()
        r = self.lreq.infer({"inputs_embeds": emb[None].astype(np.float32),
                             "attention_mask": mask.astype(np.int64),
                             "position_ids": np.arange(S, dtype=np.int64)[None],
                             "beam_idx": np.zeros(1, dtype=np.int32)})
        logits = list(r.values())[0][0]
        ttft = time.time() - t0
        nxt = int(logits[-1].argmax())
        out = [nxt]
        pos = S
        itls = []
        while nxt != EOS and len(out) < max_new:
            te1 = self.model.language_model.embed_tokens(np.array([[nxt]], dtype=np.int64))
            mask = np.concatenate([mask, [[1]]], axis=1)
            ts = time.time()
            r = self.lreq.infer({"inputs_embeds": np.array(te1).astype(np.float32),
                                 "attention_mask": mask.astype(np.int64),
                                 "position_ids": np.array([[pos]], dtype=np.int64),
                                 "beam_idx": np.zeros(1, dtype=np.int32)})
            itls.append(time.time() - ts)
            logits = list(r.values())[0][0]
            nxt = int(logits[-1].argmax())
            out.append(nxt)
            pos += 1
        return ttft, itls, len(out), S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precisions", default="fp16,int4")
    ap.add_argument("--devices", default="CPU,GPU.0")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ART, "perf_results.json"))
    args = ap.parse_args()
    os.makedirs(ART, exist_ok=True)
    fn, q = SAMPLE
    rows = []
    for prec in [p.strip() for p in args.precisions.split(",") if p.strip()]:
        ir_dir = f"ov_ir_{prec}"
        for dev in [d.strip() for d in args.devices.split(",") if d.strip()]:
            print(f"\n=== perf {prec} on {dev} ===")
            pipe = Pipe(ir_dir, dev)
            ids, am, emb = pipe.prep(fn, q)
            # warmup
            pipe.run_once(ids, am, emb)
            ttfts, itl_means, ntoks = [], [], []
            prefill_len = None
            for _ in range(args.runs):
                ttft, itls, n, S = pipe.run_once(ids, am, emb)
                ttfts.append(ttft)
                if itls:
                    itl_means.append(float(np.mean(itls)))
                ntoks.append(n)
                prefill_len = S
            row = {"precision": prec, "device": dev, "prefill_len": prefill_len,
                   "gen_tokens": int(np.median(ntoks)),
                   "ttft_s_mean": round(float(np.mean(ttfts)), 4),
                   "itl_ms_mean": round(float(np.mean(itl_means)) * 1000, 2) if itl_means else None,
                   "tok_per_s": round(1.0 / float(np.mean(itl_means)), 2) if itl_means else None,
                   "runs": args.runs}
            rows.append(row)
            print(f"  -> TTFT={row['ttft_s_mean']}s ITL={row['itl_ms_mean']}ms "
                  f"({row['tok_per_s']} tok/s) gen={row['gen_tokens']} prefill={prefill_len}")
            del pipe
    json.dump({"sample": f"{fn}::{q}", "rows": rows}, open(args.out, "w"), indent=2)
    print("\n==== PERF TABLE ====")
    print(f"{'prec':6} {'device':8} {'TTFT(s)':9} {'ITL(ms)':9} {'tok/s':8}")
    for r in rows:
        print(f"{r['precision']:6} {r['device']:8} {r['ttft_s_mean']:<9} {r['itl_ms_mean']:<9} {r['tok_per_s']}")
    print("saved", args.out)


if __name__ == "__main__":
    main()
