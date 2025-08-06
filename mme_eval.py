import string
from argparse import ArgumentParser
from pathlib import Path

import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, set_seed

from optimum.intel import OVModelForVisualCausalLM


class calculate_metrics:
    def divide_chunks(self, l, n=2):
        for i in range(0, len(l), n):
            yield l[i : i + n]
        return

    def parse_pred_ans(self, pred_ans):
        pred_ans = pred_ans.lower()
        exclude = set(string.punctuation)
        pred_ans = "".join(ch for ch in pred_ans if ch not in exclude)

        pred_label = None
        if pred_ans in ["yes", "no"]:
            pred_label = pred_ans
        else:
            prefix_pred_ans = pred_ans[:4]
            if "yes" in prefix_pred_ans:
                pred_label = "yes"
            elif "no" in prefix_pred_ans:
                pred_label = "no"
            else:
                pred_label = "other"
        return pred_label

    def compute_metric(self, gts, preds):
        assert len(gts) == len(preds)
        label_map = {"yes": 1, "no": 0, "other": -1}
        gts = [label_map[x] for x in gts]
        preds = [label_map[x] for x in preds]
        acc = accuracy_score(gts, preds)

        clean_gts, clean_preds = [], []
        other_num = 0
        for gt, pred in zip(gts, preds):
            if pred == -1:
                other_num += 1
                continue
            clean_gts.append(gt)
            clean_preds.append(pred)

        conf_mat = confusion_matrix(clean_gts, clean_preds, labels=[1, 0])
        precision = precision_score(clean_gts, clean_preds, average="binary")
        recall = recall_score(clean_gts, clean_preds, average="binary")
        tp, fn = conf_mat[0]
        fp, tn = conf_mat[1]

        return {
            "TP": tp,
            "FN": fn,
            "TN": tn,
            "FP": fp,
            "precision": precision,
            "recall": recall,
            "other_num": other_num,
            "acc": acc,
        }


def evaluate(model_path: str, model_id: str, category: str, max_new_tokens: int = 32, limit=None):
    dataset = load_dataset("darkyarding/MME", split="test")
    if category != "all":
        dataset = dataset.filter(lambda x: x["category"] == category)

    model_cls = OVModelForVisualCausalLM

    model = model_cls.from_pretrained(
        model_path,
        trust_remote_code=True,
        temperature=None,
        top_p=None,
        top_k=None,
        # ov_config={"INFERENCE_PRECISION_HINT": "f32"}
    ).eval()

    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except:
        processor = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except:
        tokenizer = None
    preprocess_kwargs = {}
    if "internvl2" in model_id.lower():
        preprocess_kwargs = {"config": AutoConfig.from_pretrained(model_id, trust_remote_code=True)}

    metric_util = calculate_metrics()

    if limit is not None:
        dataset = dataset.take(limit)

    all_items = []
    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset)):
            prompt = example["question"]
            answer = example["answer"].strip().lower()
            image = example["image"].convert("RGB")

            inputs = model.preprocess_inputs(
                text=prompt,
                image=image,
                processor=processor,
                tokenizer=tokenizer,
                **preprocess_kwargs,
            )

            generate_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]

            response = processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            pred_label = metric_util.parse_pred_ans(response)
            all_items.append((image, prompt, answer, pred_label))

    grouped = list(metric_util.divide_chunks(all_items, n=2))

    acc_plus_correct = 0
    flat_preds = []
    flat_gts = []

    for pair in grouped:
        if len(pair) != 2:
            continue
        img_correct = 0
        for _, _, gt, pred in pair:
            flat_preds.append(pred)
            flat_gts.append(gt)
            if gt == pred:
                img_correct += 1
        if img_correct == 2:
            acc_plus_correct += 1

    metrics = metric_util.compute_metric(flat_gts, flat_preds)
    acc_plus = acc_plus_correct / len(grouped)
    metrics["acc_plus"] = acc_plus

    print(f"\n MME Evaluation for '{category}'")
    with open(f"{model_path}/mme_eval_{category}.txt", "w") as f:
        f.write(f"\n MME Evaluation for '{category}'\n")
        for k, v in metrics.items():
            print(f"{k:>12}: {v:.4f}" if isinstance(v, float) else f"{k:>12}: {v}")
            f.write(f"{k:>12}: {v:.4f}\n" if isinstance(v, float) else f"{k:>12}: {v}\n")
    return metrics["acc"]


def evaluate_on_many():
    model_ids = [
        "OpenGVLab/InternVL2-1B",
        "OpenGVLab/InternVL2-4B",
        "OpenGVLab/InternVL2_5-8B",
        "microsoft/Phi-4-multimodal-instruct",
        "microsoft/Phi-3.5-vision-instruct",
        "Qwen/Qwen2-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "llava-hf/LLaVA-NeXT-Video-7B-hf",
        "llava-hf/llava-1.5-7b-hf",
        "openbmb/MiniCPM-V-2_6",  # works on transformers 4.48
    ]

    models_dir = Path("/dev/data/nsavelye/workspace/models/vlm_ptq")
    category = "scene"

    accuracies = {}
    for model_id in model_ids:
        model_parent_path = models_dir / model_id.split("/")[-1]
        for model_path in sorted(model_parent_path.glob("*/openvino_language_model.xml")):
            model_path = model_path.parent
            if (model_path / f"mme_eval_{category}.txt").exists():
                print(f"Skipping {model_path} as it has already been evaluated.")
                continue
            print("\n" + "-" * 100 + f"\nEvaluating {model_path}...")
            try:
                acc = evaluate(model_path=str(model_path), model_id=model_id, category=category)
            except Exception as e:
                # raise e
                print(f"Error evaluating {model_path}: {e}")
                continue
            accuracies[str(model_path)] = acc
            print(f"Finished evaluating {model_path}.")

    for model_path, acc in accuracies.items():
        print(f"{model_path}\taccuracy: {acc:.4f}")

    for model_id in model_ids:
        model_parent_path = models_dir / model_id.split("/")[-1]
        for model_path in sorted(model_parent_path.glob("*/openvino_language_model.xml")):
            model_path = model_path.parent
            if not (model_path / f"mme_eval_{category}.txt").exists():
                continue
            with open(model_path / f"mme_eval_{category}.txt", "r") as f:
                for line in f:
                    if "acc" in line:
                        print(model_path, line.strip())
                        break


if __name__ == "__main__":
    set_seed(42)

    # evaluate_on_many()

    eval_type_dict = [
        "existence",
        "count",
        "position",
        "color",
        "posters",
        "celebrity",
        "scene",
        "landmark",
        "artwork",
        "OCR",
    ] + ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning", "all"]

    parser = ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to ov model")
    parser.add_argument("--model-id", type=str, required=True, help="Huggingface model repo")
    parser.add_argument("--subset", choices=eval_type_dict, required=True, help="MME category name (e.g., 'Counting')")
    parser.add_argument("--limit", type=int, help="Number of samples to evaluate on", default=None)
    args = parser.parse_args()

    evaluate(model_id=args.model_id, model_path=args.model_path, category=args.subset, limit=args.limit)
