import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

import os
import torch
import string
from datasets import load_dataset
from transformers import AutoProcessor, set_seed, AutoTokenizer, AutoConfig
from argparse import ArgumentParser
from tqdm import tqdm

from optimum.intel import OVModelForVisualCausalLM


class calculate_metrics:
    def divide_chunks(self, l, n=2):
        for i in range(0, len(l), n):
            yield l[i:i + n]
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
        precision = precision_score(clean_gts, clean_preds, average='binary')
        recall = recall_score(clean_gts, clean_preds, average='binary')
        tp, fn = conf_mat[0]
        fp, tn = conf_mat[1]

        return {
            "TP": tp, "FN": fn, "TN": tn, "FP": fp,
            "precision": precision, "recall": recall,
            "other_num": other_num, "acc": acc,
        }


def evaluate(model_path: str, model_id: str, category: str, max_new_tokens: int = 512):
    dataset = load_dataset("darkyarding/MME", split="test")
    if category != "all":
        dataset = dataset.filter(lambda x: x["category"] == category)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    # model_cls = get_model_class(model_id)
    model_cls = OVModelForVisualCausalLM

    model = model_cls.from_pretrained(
        model_path,
        trust_remote_code=True,
        # torch_dtype=torch.bfloat16,
        # device_map="auto",
        # token=os.environ.get("HF_TOKEN", None),
        temperature=None,
        top_p=None,
        top_k=None,
    ).eval()

    try:
        processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    except:
        processor = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    except:
        tokenizer = None
    preprocess_kwargs = {}
    if "internvl2" in args.model_id.lower():
        preprocess_kwargs = {"config": AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)}

    metric_util = calculate_metrics()

    all_items = []
    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset)):
            # if i == 10:
            #     break
            prompt = example["question"]
            answer = example["answer"].strip().lower()
            image = example["image"].convert("RGB")

            inputs = model.preprocess_inputs(
                text=prompt, image=image, processor=processor, tokenizer=tokenizer, **preprocess_kwargs,
            )

            generate_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
            )
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]

            response = processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
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
    for k, v in metrics.items():
        print(f"{k:>12}: {v:.4f}" if isinstance(v, float) else f"{k:>12}: {v}")


if __name__ == "__main__":
    set_seed(42)

    eval_type_dict = \
        ["existence", "count", "position", "color", "posters", "celebrity", "scene", "landmark", "artwork", "OCR"] + \
        ["commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning", "all"]

    parser = ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to ov model")
    parser.add_argument("--model-id", type=str, required=True, help="Huggingface model repo")
    parser.add_argument("--subset", choices=eval_type_dict, required=True, help="MME category name (e.g., 'Counting')")
    args = parser.parse_args()

    evaluate(model_id=args.model_id, model_path=args.model_path, category=args.subset)
