# PaddleOCR-VL (`paddleocr_vl`) analysis

## Identity
- model_id: `PaddlePaddle/PaddleOCR-VL-1.5`
- model_type: `paddleocr_vl`
- architectures: `PaddleOCRVLForConditionalGeneration`
- task: image-text-to-text
- trust_remote_code: True (auto_map -> modeling_paddleocr_vl / configuration_paddleocr_vl)
- transformers authored: 4.55.0; installed 4.57.6; openvino present.

## Config invariants
- text (flat, no text_config): hidden_size=1024, num_attention_heads=16,
  num_key_value_heads=2, head_dim=128 (NOT hidden/heads=64), num_hidden_layers=18,
  intermediate_size=3072, vocab_size=103424, rope_theta=500000,
  rope_scaling.mrope_section=[16,24,24] (sum=64=head_dim/2), silu, rms_norm_eps=1e-5,
  tie_word_embeddings=False, use_bias=False.
- image_token_id=100295, video_token_id=101307, vision_start_token_id=101305,
  vision_end_token_id=101306.
- vision_config (SigLIP variant): hidden_size=1152, num_attention_heads=16,
  num_hidden_layers=27, intermediate_size=4304, patch_size=14, image_size=384,
  spatial_merge_size=2, temporal_patch_size=2, tokens_per_second=2,
  gelu_pytorch_tanh, layer_norm_eps=1e-6.

## Components (real transformers remote code)
- `PaddleOCRVLForConditionalGeneration`
  - `model`: `Ernie4_5Model` (Qwen2-VL-like text decoder w/ mrope + GQA + head_dim=128)
  - `lm_head`: Linear(1024, vocab), untied
  - `visual`: `PaddleOCRVisionModel` -> `PaddleOCRVisionTransformer`
    - `embeddings` (`PaddleOCRVisionEmbeddings`): Conv2d patch embed + interpolated
      position embedding per image grid.
    - `encoder` (`PaddleOCREncoder`): 27 x `PaddleOCREncoderLayer`, SigLIP MHA with
      2D rope over (h,w) pids; window_size=-1 in VLM path (full attention split by
      cu_seqlens); sdpa.
    - `post_layernorm`
  - `mlp_AR`: `Projector` (pre_norm LN + Linear + GELU + Linear) with a 2x2 spatial
    merge rearrange `(t h p1 w p2) d -> (t h w) (p1 p2 d)`, p1=p2=2.
- get_rope_index: identical structure to Qwen2-VL mrope 3D index (image/video/text).
- image feature insertion: masked_scatter on `input_ids == image_token_id`.

## Forward contract in VLM path (single image)
- processor keys: input_ids[1,L], attention_mask[1,L], pixel_values[N,3,14,14],
  image_grid_thw[1,3] (t,h,w). N == t*h*w. pixel_values passed to visual as 5D
  (unsqueeze(0)) -> patch_embed -> [1,N,1152].
- vision: use_rope=True, window_size=-1, vision_return_embed_list=True,
  return_pooler_output=False -> pooling head NOT used; output split by cu_seqlens.
- mlp_AR merges 2x2 -> tokens count N/4 == number of image tokens in text.

## Known incompatibility (routed failure)
- Hub `modeling_paddleocr_vl.py` calls `create_causal_mask(config=..., inputs_embeds=...)`
  (4.55 API). transformers 4.57.6 renamed the kwarg to `input_embeds` -> TypeError.
- Fix belongs in Optimum language-model patcher: wrap `create_causal_mask` in the
  model's module namespace to accept `inputs_embeds`.

## Closest supported reference
- `qwen2_vl` (optimum/exporters/openvino/model_configs.py:Qwen2VLOpenVINOConfig,
  model_patcher.py:Qwen2VLLanguageModelPatcher/Qwen2VLVisionEmbMergerPatcher,
  modeling_visual_language.py:_OVQwen2VLForCausalLM).
- Gaps vs qwen2_vl:
  - language: head_dim=128 explicit (need NormalizedConfig w/ head_dim +
    num_key_value_heads; GemmaDummyPastKeyValuesGenerator honours head_dim), and the
    create_causal_mask compat fix.
  - vision: SigLIP-variant tower (Conv2d patch embed + interpolated pos embed + 2D
    rope + post_layernorm + 2x2 Projector), not the Qwen ViT. Data-dependent
    control flow (per-image interpolation loop, pids, cu_seqlens split, einops
    rearrange, dynamic pooling head unused in VLM path).

## Integration plan / files
- optimum/exporters/openvino/model_configs.py: PaddleOCRVLConfigBehavior enum,
  PaddleOCRVLOpenVINOConfig (register paddleocr_vl), internal Ernie language config,
  vision/merger inputs/outputs.
- optimum/exporters/openvino/input_generators.py: dummy generators for LM
  (position_ids 3D), vision embeddings and merger.
- optimum/exporters/openvino/model_patcher.py: PaddleOCRVLLanguageModelPatcher
  (create_causal_mask fix + forward wrap), vision embeddings + merger patchers to
  make the tower traceable (precompute pos-embed & rope in runtime, pass as inputs).
- optimum/intel/openvino/modeling_visual_language.py: _OVPaddleOCRVLForCausalLM
  (get_rope_index, get_vision_embeddings, get_multimodal_embeddings, preprocess).
- tests + docs.

## Version bounds
- MIN_TRANSFORMERS_VERSION follows qwen2_vl runtime (>=4.57); MAX "5.0".
