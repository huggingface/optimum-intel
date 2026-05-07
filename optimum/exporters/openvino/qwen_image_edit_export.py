"""Custom OpenVINO export support for Qwen/Qwen-Image-Edit."""

import gc
import logging
import time
from pathlib import Path
from typing import Union

import torch
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model if hasattr(model, "model") else model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=False,
        )
        return outputs[0] if isinstance(outputs, (tuple, list)) else outputs.last_hidden_state


class VAEEncoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        posterior = self.vae.encode(x, return_dict=False)[0]
        return posterior.mode() if hasattr(posterior, "mode") else posterior


class VAEDecoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        outputs = self.vae.decode(z, return_dict=False)
        return outputs[0] if isinstance(outputs, tuple) else outputs.sample


class TransformerWrapper(torch.nn.Module):
    def __init__(self, model, img_shapes):
        super().__init__()
        self.model = model
        self.img_shapes = img_shapes

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            timestep=timestep,
            img_shapes=self.img_shapes,
            return_dict=False,
        )
        return outputs[0] if isinstance(outputs, (tuple, list)) else outputs.sample


def _try_ov_export(model, example_input):
    import openvino as ov

    return ov.convert_model(model, example_input=example_input)


def _try_torch_export(model, example_input):
    import openvino as ov

    with torch.no_grad():
        if isinstance(example_input, dict):
            exported = torch.export.export(model, args=(), kwargs=example_input, strict=False)
        elif isinstance(example_input, tuple):
            exported = torch.export.export(model, args=example_input, strict=False)
        else:
            exported = torch.export.export(model, args=(example_input,), strict=False)
    return ov.convert_model(exported)


def _try_jit_trace(model, example_input):
    import openvino as ov

    with torch.no_grad():
        if isinstance(example_input, dict):
            keys = list(example_input.keys())
            values = tuple(example_input.values())

            class _DictWrapper(torch.nn.Module):
                def __init__(self, wrapped, names):
                    super().__init__()
                    self.wrapped = wrapped
                    self.names = names

                def forward(self, *args):
                    return self.wrapped(**dict(zip(self.names, args)))

            traced = torch.jit.trace(_DictWrapper(model, keys), values, strict=False)
            return ov.convert_model(traced, example_input=values)
        if isinstance(example_input, tuple):
            traced = torch.jit.trace(model, example_input, strict=False)
            return ov.convert_model(traced, example_input=example_input)
        traced = torch.jit.trace(model, (example_input,), strict=False)
        return ov.convert_model(traced, example_input=(example_input,))


def _export_component(model, example_input, out_path: Path, name: str) -> bool:
    import openvino as ov

    for strategy_name, strategy in (
        ("ov.convert_model", _try_ov_export),
        ("torch.export", _try_torch_export),
        ("torch.jit.trace", _try_jit_trace),
    ):
        logger.info("[%s] Trying %s", name, strategy_name)
        try:
            start = time.time()
            ov_model = strategy(model, example_input)
            ov.save_model(ov_model, str(out_path))
            logger.info(
                "[%s] SUCCESS via %s in %.1fs (%.1f MB)",
                name,
                strategy_name,
                time.time() - start,
                out_path.stat().st_size / 1e6,
            )
            return True
        except Exception as error:
            logger.warning("[%s] %s failed: %s: %s", name, strategy_name, type(error).__name__, str(error)[:300])

    logger.error("[%s] All strategies failed", name)
    return False


def _load_components(model_id: str, dtype: torch.dtype):
    from diffusers import AutoencoderKLQwenImage, QwenImageTransformer2DModel
    from transformers import Qwen2_5_VLForConditionalGeneration

    model_dir = Path(snapshot_download(model_id))
    logger.info("Loading components from %s", model_dir)

    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir / "text_encoder",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_dir / "transformer",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    vae = AutoencoderKLQwenImage.from_pretrained(
        model_dir / "vae",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    return text_encoder, transformer, vae


def export_qwen_image_edit(
    model_id: str = "Qwen/Qwen-Image-Edit",
    output_dir: Union[str, Path] = "./ov_model_qwen_image_edit",
    dtype: torch.dtype = torch.bfloat16,
    image_size: int = 512,
    export_text_encoder: bool = True,
    export_vae: bool = True,
    export_transformer: bool = True,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    text_encoder, transformer, vae = _load_components(model_id, dtype)

    if export_text_encoder:
        text_encoder = text_encoder.eval().to(torch.float32)
        text_encoder_wrapper = TextEncoderWrapper(text_encoder).eval()
        text_inputs = {
            "input_ids": torch.ones((1, 32), dtype=torch.long),
            "attention_mask": torch.ones((1, 32), dtype=torch.long),
        }
        with torch.no_grad():
            text_encoder_wrapper(**text_inputs)
        results["text_encoder"] = _export_component(
            text_encoder_wrapper,
            text_inputs,
            output_dir / "text_encoder.xml",
            "text_encoder",
        )
        del text_encoder_wrapper, text_encoder
        gc.collect()

    if export_vae:
        vae = vae.eval().to(torch.float32)
        vae_encoder = VAEEncoderWrapper(vae).eval()
        image = torch.randn(1, 3, 1, image_size, image_size)
        with torch.no_grad():
            latent = vae_encoder(image)
        results["vae_encoder"] = _export_component(vae_encoder, image, output_dir / "vae_encoder.xml", "vae_encoder")

        vae_decoder = VAEDecoderWrapper(vae).eval()
        results["vae_decoder"] = _export_component(
            vae_decoder,
            torch.randn_like(latent),
            output_dir / "vae_decoder.xml",
            "vae_decoder",
        )
        del vae_decoder, vae_encoder, vae
        gc.collect()

    if export_transformer:
        transformer = transformer.eval().to(torch.float32)
        patch_size = getattr(transformer.config, "patch_size", 2)
        height = image_size // 8 // patch_size
        width = image_size // 8 // patch_size
        seq_len = height * width
        transformer_wrapper = TransformerWrapper(transformer, [(1, height, width)]).eval()
        transformer_inputs = {
            "hidden_states": torch.randn(1, seq_len, transformer.config.in_channels),
            "encoder_hidden_states": torch.randn(1, 32, transformer.config.joint_attention_dim),
            "encoder_hidden_states_mask": torch.ones(1, 32, dtype=torch.bool),
            "timestep": torch.tensor([500.0]),
        }
        with torch.no_grad():
            transformer_wrapper(**transformer_inputs)
        results["transformer"] = _export_component(
            transformer_wrapper,
            transformer_inputs,
            output_dir / "transformer.xml",
            "transformer",
        )
        del transformer_wrapper, transformer
        gc.collect()

    return results


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    destination = sys.argv[1] if len(sys.argv) > 1 else "./ov_model_qwen_image_edit"
    summary = export_qwen_image_edit(output_dir=destination)
    raise SystemExit(0 if any(summary.values()) else 1)
