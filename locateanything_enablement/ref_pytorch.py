#!/usr/bin/env python3
"""PyTorch reference for nvidia/LocateAnything-3B prefill + slow-AR greedy.

Captures prefill logits and a short greedy AR token sequence on one sample, and
saves the exact processed inputs so the OpenVINO export can be compared 1:1.
"""
import json
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

MODEL_ID = "nvidia/LocateAnything-3B"
OUT = "parity_artifacts"
QUERY = "Locate all the instances that matches the following description: person."
RES = 448
N_GREEDY = 8

os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)


def make_image():
    # Deterministic synthetic RGB image at fixed resolution (content irrelevant for parity).
    rng = np.random.RandomState(0)
    base = np.linspace(0, 255, RES, dtype=np.float32)
    img = np.stack(
        [
            np.tile(base, (RES, 1)),
            np.tile(base[:, None], (1, RES)),
            (rng.rand(RES, RES) * 255).astype(np.float32),
        ],
        axis=-1,
    )
    # add a couple of solid rectangles
    img[80:200, 100:260, 0] = 30
    img[250:360, 300:420, 1] = 220
    return Image.fromarray(img.clip(0, 255).astype(np.uint8), "RGB")


def main():
    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=dtype, trust_remote_code=True).eval()

    # Force tie_weights (model __init__ skips post_init()).
    model.tie_weights()
    print("text attn impl:", model.config.text_config._attn_implementation)
    print("vision attn impl:", model.config.vision_config._attn_implementation)

    image = make_image()
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": QUERY}]}]
    text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = processor.process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt")

    pixel_values = inputs["pixel_values"].to(dtype)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    image_grid_hws = inputs.get("image_grid_hws")
    if isinstance(image_grid_hws, np.ndarray):
        image_grid_hws = torch.from_numpy(image_grid_hws)
    image_grid_hws = image_grid_hws.to(torch.int32)

    print(
        "pixel_values",
        tuple(pixel_values.shape),
        "grid",
        image_grid_hws.tolist(),
        "input_ids",
        tuple(input_ids.shape),
        "num_image_tokens",
        int((input_ids == model.config.image_token_index).sum()),
    )

    @torch.no_grad()
    def vision_feats():
        vit = model.extract_feature(pixel_values, image_grid_hws)
        if isinstance(vit, (list, tuple)):
            vit = torch.cat(vit, dim=0)
        return model.mlp1(vit)  # (L_post, 2048)

    # Cache the projected vision features once (resolution is fixed).
    _feats = vision_feats()

    @torch.no_grad()
    def prefill_logits(ids, amask):
        # Pass input_ids + visual_features so the vendored Qwen2 takes its
        # standard-causal (slow-AR) mask branch; image_processing scatters the
        # projected features into image_token positions internally.
        out = model.language_model(
            input_ids=ids,
            visual_features=_feats.to(model.language_model.dtype),
            image_token_index=model.config.image_token_index,
            attention_mask=amask,
            use_cache=False,
        )
        return out.logits

    # ---- prefill logits ----
    logits = prefill_logits(input_ids, attention_mask).float()
    print("prefill logits", tuple(logits.shape))
    np.save(os.path.join(OUT, "ref_prefill_logits.npy"), logits[0].numpy())

    # ---- slow-AR greedy (recompute, no cache) ----
    gen = input_ids.clone()
    amask = attention_mask.clone()
    greedy = []
    for _ in range(N_GREEDY):
        lg = prefill_logits(gen, amask)
        nxt = int(lg[0, -1].argmax())
        greedy.append(nxt)
        gen = torch.cat([gen, torch.tensor([[nxt]], dtype=gen.dtype)], dim=1)
        amask = torch.cat([amask, torch.ones((1, 1), dtype=amask.dtype)], dim=1)
    print("greedy tokens:", greedy)
    print("greedy decoded:", tokenizer.decode(greedy))

    # ---- persist inputs for OV ----
    np.save(os.path.join(OUT, "pixel_values.npy"), pixel_values.float().numpy())
    np.save(os.path.join(OUT, "image_grid_hws.npy"), image_grid_hws.numpy())
    np.save(os.path.join(OUT, "input_ids.npy"), input_ids.numpy())
    np.save(os.path.join(OUT, "attention_mask.npy"), attention_mask.numpy())
    np.save(os.path.join(OUT, "ref_vision_feats.npy"), vision_feats().float().numpy())
    with open(os.path.join(OUT, "ref_meta.json"), "w") as f:
        json.dump(
            {
                "query": QUERY,
                "resolution": RES,
                "grid": image_grid_hws.tolist(),
                "input_ids_len": int(input_ids.shape[1]),
                "num_image_tokens": int((input_ids == model.config.image_token_index).sum()),
                "greedy_tokens": greedy,
                "greedy_text": tokenizer.decode(greedy),
                "image_token_index": int(model.config.image_token_index),
            },
            f,
            indent=2,
        )
    print("saved reference artifacts to", OUT)


if __name__ == "__main__":
    main()
