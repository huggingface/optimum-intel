import hashlib
from pathlib import Path

import aiohttp
import numpy as np
import requests
from PIL import Image
from datasets import load_dataset
from matplotlib import pyplot as plt
from scipy.ndimage import center_of_mass
from tqdm import tqdm
from transformers import SamModel, SamProcessor

from optimum.intel import OVWeightQuantizationConfig, OVPipelineQuantizationConfig
from optimum.intel.openvino import OVSamModel


SAVE_DIR = Path("sam")

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask[0].reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def sample_points_from_mask(mask: np.ndarray, num_points: int = 1, seed_offset: int = 0):
    if not mask.any():
        return []  # No valid points

    # Use the mask's hash as a seed
    mask_bytes = mask.astype(np.uint8).tobytes()
    mask_hash = hashlib.md5(mask_bytes).hexdigest()
    seed = int(mask_hash, 16) + seed_offset
    rng = np.random.default_rng(seed)

    # Get all True indices
    coords = np.argwhere(mask)

    if len(coords) < num_points:
        sampled = coords
    else:
        sampled = coords[rng.choice(len(coords), size=num_points, replace=False)]
    sampled = [[it[1], it[0]] for it in sampled]  # Convert to (x, y) format
    # sampled = [[it[0], it[1]] for it in sampled]

    return sampled


def load_model(backend, model_id, label=None, quantization_config=None):
    if backend == "ov":
        if label is None:
            label = "tmp"
        load_dir = SAVE_DIR / model_id.split("/")[-1] / label
        model = None
        try:
            model = OVSamModel.from_pretrained(load_dir)
        except Exception as e:
            pass
        if model is None:
            model = OVSamModel.from_pretrained(
                model_id,
                load_in_8bit=False,
                quantization_config=quantization_config,
            )
            model.save_pretrained(load_dir)
    else:
        device = "cuda"
        # device = "cpu"
        model = SamModel.from_pretrained(model_id).to(device)
    processor = SamProcessor.from_pretrained(model_id)

    return model, processor


def infer_model(model, processor: SamProcessor, image, input_points=None, input_boxes=None, input_labels=None):
    inputs = processor(
        image, input_points=input_points, input_boxes=input_boxes, input_labels=input_labels, return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # outputs = model(**inputs, multimask_output=False)
    outputs = model(**inputs)
    pred_mask = outputs.pred_masks[:, :, int(outputs.iou_scores.argmax())][None]
    masks = processor.image_processor.post_process_masks(
        pred_mask.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    return masks[0]


def demo_prediction(model, processor, save_filename):
    input_boxes = input_points = None

    # img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    # image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    # input_points = [[[450, 600]]]

    dataset = load_dataset(
        "scene_parse_150", "instance_segmentation", split="validation",
        trust_remote_code=True, storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}}
    )
    it = 2
    image = dataset[it]["image"]
    annotation = dataset[it]["annotation"]
    annotation = np.array(annotation.convert("RGB"))
    category_mask = annotation[:, :, 0]
    instance_mask = annotation[:, :, 1]
    input_masks = []
    # input_boxes = []
    input_points = []
    input_labels = []
    start_id = 0
    for instance_id in range(start_id, int(instance_mask.max())):
        mask = instance_mask == instance_id + 1
        assert np.any(mask)
        # y, x = np.where(mask)
        input_masks.append(mask)
        # input_points.append([list(center_of_mass(mask))])
        positive_points = sample_points_from_mask(mask, num_points=max(10, int(np.sqrt(mask.sum()) * 0.2)))
        negative_points = sample_points_from_mask(~mask, num_points=max(10, int(np.sqrt((~mask).sum()) * 0.2)))
        input_points.append(positive_points + negative_points)
        input_labels.append([[1] * len(positive_points) + [-1] * len(negative_points)])
        # input_boxes.append([np.min(x), np.min(y), np.max(x), np.max(y)])
        # break
    # input_boxes = [input_boxes]
    # input_points = [input_points]

    # output_masks = infer_model(model, processor, [image]*len(input_points), input_points=input_points, input_labels=input_labels)

    # output_masks = np.array([[input_masks]])
    output_masks = []
    for points, labels in tqdm(zip(input_points, input_labels), total=len(input_points)):
        mask = infer_model(model, processor, image, input_points=[points], input_labels=labels)
        output_masks.append(mask[0])
    plt.imshow(np.array(image))
    ax = plt.gca()
    for mask in output_masks:
        show_mask(mask, ax=ax, random_color=True)
    plt.axis("off")
    plt.savefig(save_filename, bbox_inches="tight", pad_inches=0)
    plt.cla()
    plt.clf()

def validate_on_dataset(model, processor, dataset_size):
    dataset = load_dataset(
        "scene_parse_150", "instance_segmentation", split="validation",
        trust_remote_code=True, storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}}
    )
    dataset = dataset.shuffle(seed=0)

    iou_scores = []
    for i in tqdm(range(dataset_size)):
        image = dataset[i]["image"]
        annotation = dataset[i]["annotation"]
        annotation = np.array(annotation.convert("RGB"))
        category_mask = annotation[:, :, 0]
        instance_mask = annotation[:, :, 1]
        input_masks = []
        input_points = []
        input_labels = []
        start_id = 0
        for instance_id in range(start_id, int(instance_mask.max())):
            mask = instance_mask == instance_id + 1
            assert np.any(mask)
            input_masks.append(mask)
            # positive_points = [list(center_of_mass(mask))]
            # negative_points = []
            positive_points = sample_points_from_mask(mask, num_points=max(10, int(np.sqrt(mask.sum()) * 0.2)))
            negative_points = sample_points_from_mask(~mask, num_points=max(10, int(np.sqrt((~mask).sum()) * 0.2)))
            input_points.append(positive_points + negative_points)
            input_labels.append([[1] * len(positive_points) + [-1] * len(negative_points)])
            # break

        output_masks = []
        for points, labels in zip(input_points, input_labels):
            mask = infer_model(model, processor, image, input_points=[points], input_labels=labels)
            output_masks.append(mask[0])

        for input_mask, output_mask in zip(input_masks, output_masks):
            intersection = np.logical_and(input_mask, output_mask).sum()
            union = np.logical_or(input_mask, output_mask).sum()
            if union == 0:
                iou = 1.0 if intersection == 0 else 0.0
            else:
                iou = intersection / union
            iou_scores.append(iou)
    mean_iou = np.mean(iou_scores)
    print(f"Mean IoU over {len(iou_scores)} masks: {mean_iou:.4f}")
    return mean_iou

if __name__ == "__main__":
    model_id = "facebook/sam-vit-base"
    for model_id in [
        "facebook/sam-vit-base",
        # "facebook/sam-vit-large",
        # "facebook/sam-vit-huge",
    ]:
        quantization_config = None
        # pt_model, processor = load_model("pt", model_id)
        # ov_model_fp32, processor = load_model("ov", model_id, "ov_fp32")
        # ov_model_int8, processor = load_model("ov", model_id, "ov_w_int8", OVWeightQuantizationConfig(bits=8))
        ov_model_int8, processor = load_model(
            "ov", model_id, "ov_ve-w-int8", OVPipelineQuantizationConfig(
                {
                    "vision_encoder_model": OVWeightQuantizationConfig(bits=8),
                    # "prompt_encoder_mask_decoder_model": OVWeightQuantizationConfig(bits=8),
                },
            )
        )

        # demo_prediction(pt_model, processor, SAVE_DIR / "pt.png")
        # demo_prediction(ov_model_fp32, processor, SAVE_DIR / "ov_fp32.png")
        # demo_prediction(ov_model_int8, processor, SAVE_DIR / "ov_int8.png")

        # validate_on_dataset(pt_model, processor, dataset_size=100)
        # validate_on_dataset(ov_model_fp32, processor, dataset_size=100)
        validate_on_dataset(ov_model_int8, processor, dataset_size=100)
