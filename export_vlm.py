import gc
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import requests
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, AutoConfig

from optimum.exporters.openvino.__main__ import maybe_convert_tokenizers
from optimum.exporters.openvino.utils import load_preprocessors
from optimum.intel import OVPipelineQuantizationConfig, OVWeightQuantizationConfig, OVQuantizationConfig, \
    OVModelForVisualCausalLM

model_ids = [
    ("OpenGVLab/InternVL2-1B", None),
    ("OpenGVLab/InternVL2-4B", None),
    ("OpenGVLab/InternVL2_5-8B", None),
    ("microsoft/Phi-4-multimodal-instruct", "refs/pr/78"),
    ("microsoft/Phi-3.5-vision-instruct", "refs/pr/37"),
    ("Qwen/Qwen2-VL-7B-Instruct", None),
    ("Qwen/Qwen2.5-VL-3B-Instruct", None),
    ("llava-hf/LLaVA-NeXT-Video-7B-hf", None),
    ("llava-hf/llava-1.5-7b-hf", None),
    # ("openbmb/MiniCPM-V-2_6", None),    # works on transformers 4.48
    # ("qnguyen3/nanoLLaVA", "refs/pr/10"),   # for some reason fails on wwb
    # ("HuggingFaceTB/SmolVLM2-256M-Video-Instruct", None),   # video based?
    # ("HuggingFaceTB/SmolVLM2-2.2B-Instruct", None),
    # ("HuggingFaceTB/SmolVLM2-256M-Video-Instruct", None),
]

wwb_path = "/home/nsavel/venvs/wwb/bin/wwb"

quantization_configs = [
    (None, "bf16"),
    (OVWeightQuantizationConfig(bits=4), "int4_w-int8"),
    (
        OVPipelineQuantizationConfig(
            {
                "lm_model": OVWeightQuantizationConfig(bits=4),
                "vision_embeddings_model": OVQuantizationConfig(bits=8, dataset="contextual", trust_remote_code=True),
                "other": OVWeightQuantizationConfig(bits=8, sym=True),
            }
        ),
        "int4_wa-int8-w-int8"
    ),
    # (
    #     OVPipelineQuantizationConfig(
    #         {
    #             "lm_model": OVWeightQuantizationConfig(bits=4),
    #             "vision_embeddings_model": OVQuantizationConfig(bits=8, dataset="contextual", trust_remote_code=True),
    #             "vision_embeddings_merger_model": OVQuantizationConfig(bits=8, dataset="contextual", trust_remote_code=True),
    #             "other": OVWeightQuantizationConfig(bits=8, sym=True),
    #         }
    #     ),
    #     "int4_wa-int8-w-int8-merger"
    # ),
    # (OVWeightQuantizationConfig(bits=8), "int8_w-int8"),
    # (
    #     OVPipelineQuantizationConfig(
    #         {
    #             "lm_model": OVWeightQuantizationConfig(bits=8),
    #             "vision_embeddings_model": OVQuantizationConfig(bits=8, dataset="contextual", trust_remote_code=True),
    #             "other": OVWeightQuantizationConfig(bits=8, sym=True),
    #         }
    #     ),
    #     "int8_wa-int8-w-int8"
    # ),
]

# for q_config, label in quantization_configs:
#     for model_id, revision in model_ids:
#         print("\nExporting model:", model_id, "with config:", label); print()
#         save_dir = "/home/nsavel/workspace/models/vlm_ptq/" + model_id.split("/")[-1] + "/" + label
#         try:
#             trust_remote_code = True
#             model = OVModelForVisualCausalLM.from_pretrained(
#                 model_id,
#                 quantization_config=q_config,
#                 load_in_8bit=False,
#                 trust_remote_code=trust_remote_code,
#                 revision=revision,
#             )
#             model.save_pretrained(save_dir)
#
#             try:
#                 AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code).save_pretrained(save_dir)
#             except:
#                 pass
#             try:
#                 AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code).save_pretrained(save_dir)
#             except:
#                 pass
#
#             model_type = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code).model_type
#             preprocessors = load_preprocessors(
#                 model_id, subfolder="", trust_remote_code=trust_remote_code, model_type=model_type
#             )
#             maybe_convert_tokenizers("transformers", save_dir, model, preprocessors)
#         except Exception as e:
#             # raise e
#             print(f"Failed to export {model_id} with config {label}: {e}")
#             continue


# for q_config, label in quantization_configs:
#     for model_id, revision in model_ids:
#         save_dir = Path("/home/nsavel/workspace/models/vlm_ptq/" + model_id.split("/")[-1]) / label
#         print("\nValidating model:", model_id, "with config:", label); print()
#         base_model_path = save_dir.parent / "pt" if (save_dir.parent / "pt").exists() else model_id
#         wwb_command = f"{wwb_path} --base-model {base_model_path} --gt-data {save_dir.parent}/gt.csv --model-type visual-text --hf --device cuda"
#         # wwb_command = f"{wwb_path} --target-model {save_dir} --gt-data {save_dir.parent}/gt.csv --output {save_dir}/similarity --model-type visual-text"
#         print(wwb_command)
#         process = subprocess.Popen(wwb_command, shell=True)
#         process.wait()
#     break


# for model_id, revision in model_ids:
#     for q_config, label in quantization_configs:
#         save_dir = Path("/home/nsavel/workspace/models/vlm_ptq/" + model_id.split("/")[-1]) / label
#         similarity_dir = save_dir / "similarity"
#         similarity_file = similarity_dir / "metrics.csv"
#         with open(similarity_file, "r") as f:
#            contents = f.read()
#            similarity = float(contents.split(",")[-1])
#         print(f"Model: {model_id}, Config: {label}, Similarity: {similarity:.4f}")
#     print()


n_reps = 3
results = defaultdict(list)
image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
raw_image = Image.open(requests.get(image_file, stream=True).raw)
for model_id, revision in model_ids:
    print(model_id)
    for q_config, label in quantization_configs:
        save_dir = Path("/home/nsavel/workspace/models/vlm_ptq/" + model_id.split("/")[-1]) / label
        model = OVModelForVisualCausalLM.from_pretrained(save_dir, trust_remote_code=True)
        try:
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        except:
            processor = None

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        except:
            tokenizer = None
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        preprocess_kwargs = {}
        if "internvl2" in model_id.lower():
            preprocess_kwargs = {"config": config}
        inputs = model.preprocess_inputs(
            text="What are these?", image=raw_image, processor=processor, tokenizer=tokenizer, **preprocess_kwargs
        )

        generation_kwargs = {"max_new_tokens": 50, "do_sample": False, "eos_token_id": -1}
        for _ in range(n_reps):
            start_time = time.perf_counter()
            output_ids = model.generate(**inputs, **generation_kwargs)
            end_time = time.perf_counter()
            results[(model_id, label)].append(end_time - start_time)

            output_ids = output_ids[0, inputs["input_ids"].shape[1]:]
            # print(processor.decode(output_ids, skip_special_tokens=True))
            assert len(output_ids) == generation_kwargs["max_new_tokens"]
        del model
        gc.collect()
        # break
for (model_id, label), times in results.items():
    avg_time = sum(times) / len(times)
    print(f"Model: {model_id}, Config: {label}, Average Inference Time: {avg_time:.4f} seconds")
