#!/usr/bin/env python3
"""Milestone B parity: OpenVINO IR vs PyTorch reference for LocateAnything-3B.

Loads the exported IRs via optimum-intel's OVModelForVisualCausalLM and compares
prefill logits + greedy slow-AR tokens against the PyTorch reference artifacts.
"""
import json
import os

import numpy as np
import torch

OV_DIR = "ov_ir_fp16"
ART = "parity_artifacts"
N_GREEDY = 8


def cossim(a, b, axis=-1, eps=1e-8):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    num = (a * b).sum(axis)
    den = np.linalg.norm(a, axis=axis) * np.linalg.norm(b, axis=axis) + eps
    return num / den


def main():
    from optimum.intel.openvino import OVModelForVisualCausalLM

    pixel_values = np.load(os.path.join(ART, "pixel_values.npy"))
    image_grid_hws = np.load(os.path.join(ART, "image_grid_hws.npy"))
    input_ids = np.load(os.path.join(ART, "input_ids.npy"))
    attention_mask = np.load(os.path.join(ART, "attention_mask.npy"))
    ref_logits = np.load(os.path.join(ART, "ref_prefill_logits.npy"))  # (S, V)
    ref_feats = np.load(os.path.join(ART, "ref_vision_feats.npy"))  # (256, 2048)
    meta = json.load(open(os.path.join(ART, "ref_meta.json")))
    img_tok = meta["image_token_index"]

    model = OVModelForVisualCausalLM.from_pretrained(OV_DIR, trust_remote_code=True)
    print("loaded OV model:", type(model).__name__)

    # ---- vision IR parity (MoonViT + mlp1) ----
    vreq = model.vision_embeddings.request
    vout = vreq({"pixel_values": pixel_values.astype(np.float32), "image_grid_hws": image_grid_hws.astype(np.int32)})
    ov_feats = list(vout.values())[0]
    vcos = cossim(ov_feats.reshape(-1), ref_feats.reshape(-1))
    vper = cossim(ov_feats, ref_feats, axis=-1)
    print(f"vision feats shape {ov_feats.shape} ref {ref_feats.shape}")
    print(f"vision cossim flat={float(vcos):.6f} per-token mean={float(vper.mean()):.6f} min={float(vper.min()):.6f}")

    # ---- build inputs_embeds: text embeds with image features scattered ----
    inputs_embeds = model.language_model.embed_tokens(input_ids.astype(np.int64))  # (1, S, 2048)
    emb = np.array(inputs_embeds)[0].copy()
    flat_ids = input_ids[0]
    sel = flat_ids == img_tok
    assert sel.sum() == ov_feats.shape[0], (int(sel.sum()), ov_feats.shape)
    emb[sel] = ov_feats.astype(emb.dtype)
    inputs_embeds = emb[None]

    # ---- language IR prefill ----
    S = input_ids.shape[1]
    position_ids = np.arange(S, dtype=np.int64)[None]
    lreq = model.language_model.request
    lreq.reset_state()
    res = lreq.infer(
        {
            "inputs_embeds": inputs_embeds.astype(np.float32),
            "attention_mask": attention_mask.astype(np.int64),
            "position_ids": position_ids,
            "beam_idx": np.zeros(1, dtype=np.int32),
        }
    )
    ov_logits = list(res.values())[0][0]  # (S, V)
    print(f"OV prefill logits {ov_logits.shape} ref {ref_logits.shape}")

    pc = cossim(ov_logits, ref_logits, axis=-1)
    print(f"PREFILL cossim mean={float(pc.mean()):.6f} min={float(pc.min()):.6f}")
    last_cos = float(cossim(ov_logits[-1], ref_logits[-1]))
    print(f"last-position cossim={last_cos:.6f}")
    print("OV argmax last:", int(ov_logits[-1].argmax()), "ref argmax last:", int(ref_logits[-1].argmax()))

    # ---- greedy slow-AR (recompute, no cache reuse: rerun prefill each step) ----
    ov_greedy = []
    cur_ids = input_ids.copy()
    cur_mask = attention_mask.copy()
    for _ in range(N_GREEDY):
        Sc = cur_ids.shape[1]
        pos = np.arange(Sc, dtype=np.int64)[None]
        te = model.language_model.embed_tokens(cur_ids.astype(np.int64))
        e = np.array(te)[0].copy()
        s2 = cur_ids[0] == img_tok
        e[s2] = ov_feats.astype(e.dtype)
        lreq.reset_state()
        r = lreq.infer(
            {
                "inputs_embeds": e[None].astype(np.float32),
                "attention_mask": cur_mask.astype(np.int64),
                "position_ids": pos,
                "beam_idx": np.zeros(1, dtype=np.int32),
            }
        )
        lg = list(r.values())[0][0]
        nxt = int(lg[-1].argmax())
        ov_greedy.append(nxt)
        cur_ids = np.concatenate([cur_ids, [[nxt]]], axis=1)
        cur_mask = np.concatenate([cur_mask, [[1]]], axis=1)

    ref_greedy = meta["greedy_tokens"]
    match = sum(int(a == b) for a, b in zip(ov_greedy, ref_greedy))
    print("OV greedy :", ov_greedy)
    print("ref greedy:", ref_greedy)
    print(f"token match {match}/{len(ref_greedy)}")

    out = {
        "vision_cossim_flat": float(vcos),
        "vision_cossim_pertoken_mean": float(vper.mean()),
        "vision_cossim_pertoken_min": float(vper.min()),
        "prefill_cossim_mean": float(pc.mean()),
        "prefill_cossim_min": float(pc.min()),
        "last_pos_cossim": last_cos,
        "ov_greedy": ov_greedy,
        "ref_greedy": ref_greedy,
        "token_match": f"{match}/{len(ref_greedy)}",
        "resolution": meta["resolution"],
    }
    with open(os.path.join(ART, "parity_result.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved", os.path.join(ART, "parity_result.json"))


if __name__ == "__main__":
    main()
