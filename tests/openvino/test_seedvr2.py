import shutil
import sys
import tempfile
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from torchvision.io import write_png
from transformers import PretrainedConfig

from optimum.exporters.openvino.convert import export
from optimum.exporters.openvino.model_configs import SeedVR2NaDiTOpenVINOConfig
from optimum.intel import OVSeedVR2Pipeline


class NaDiTOutput:
    def __init__(self, vid_sample):
        self.vid_sample = vid_sample


class TinySeedVR2NaDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 2)
        self.config = PretrainedConfig(model_type="seedvr2", export_model_type="seedvr2", vid_in_channels=4, txt_in_dim=8)

    def forward(self, vid, txt, vid_shape, txt_shape, timestep, disable_cache=False):
        return NaDiTOutput(self.proj(vid))


def test_seedvr2_pipeline_forward_and_cfg():
    root = Path(tempfile.mkdtemp())
    try:
        model = TinySeedVR2NaDiT()
        ov_config = SeedVR2NaDiTOpenVINOConfig(model.config, task="semantic-segmentation")
        export(
            model,
            ov_config,
            root / "openvino_seedvr2_nadit.xml",
            input_shapes={"batch_size": 1, "sequence_length": 3, "num_frames": 1, "height": 2, "width": 2},
            stateful=False,
            library_name="transformers",
        )
        model.config.save_pretrained(root)

        pipeline = OVSeedVR2Pipeline.from_pretrained(root, compile=True, device="CPU")
        vid = torch.zeros((4, 4), dtype=torch.float32)
        txt = torch.zeros((3, 8), dtype=torch.float32)
        vid_shape = torch.tensor([[1, 2, 2]], dtype=torch.long)
        txt_shape = torch.tensor([[3]], dtype=torch.long)
        timestep = torch.tensor([1.0], dtype=torch.float32)

        outputs = pipeline(vid=vid, txt=txt, vid_shape=vid_shape, txt_shape=txt_shape, timestep=timestep)
        assert tuple(outputs.vid_sample.shape) == (4, 2)

        cfg = pipeline.classifier_free_guidance(
            vid[:, :2], vid[:, :2], txt, txt, vid_shape, txt_shape, txt_shape, timestep, guidance_scale=1.5
        )
        assert tuple(cfg.shape) == (4, 2)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_seedvr2_ada_modulation_patch_is_gpu_safe_and_equivalent():
    # The exporter rewrites AdaSingle.forward to read modulation components with a flat index_select
    # instead of the reshape->inner-unbind pattern the Intel GPU plugin miscomputes. The rewrite must
    # stay numerically identical to the original (single-video export, batch=1).
    import types

    from optimum.exporters.openvino.seedvr import _patch_seedvr_ada_modulation

    def expand_dims(x, dim, ndim):
        return x.reshape(x.shape[:dim] + (1,) * (ndim - x.ndim) + x.shape[dim:])

    class AdaSingle(torch.nn.Module):
        def __init__(self, dim, layers):
            super().__init__()
            self.dim = dim
            self.layers = layers
            for name in layers:
                self.register_parameter(f"{name}_shift", torch.nn.Parameter(torch.randn(dim)))
                self.register_parameter(f"{name}_scale", torch.nn.Parameter(torch.randn(dim) + 1))
                self.register_parameter(f"{name}_gate", torch.nn.Parameter(torch.randn(dim)))

        def forward(self, hid, emb, layer, mode, cache=None, branch_tag="", hid_len=None):
            idx = self.layers.index(layer)
            emb = emb.reshape(emb.shape[0], self.dim, len(self.layers), 3)[..., idx, :]
            emb = expand_dims(emb, 1, hid.ndim + 1)
            if hid_len is not None:
                emb = torch.cat([element.repeat(int(length), *([1] * element.ndim)) for element, length in zip(emb, hid_len)])
            shiftA, scaleA, gateA = emb.unbind(-1)
            if mode == "in":
                return hid * (scaleA + getattr(self, f"{layer}_scale")) + (shiftA + getattr(self, f"{layer}_shift"))
            return hid * (gateA + getattr(self, f"{layer}_gate"))

    module = types.ModuleType("_seedvr_ada_test")
    module.AdaSingle = AdaSingle
    module.expand_dims = expand_dims
    sys.modules["_seedvr_ada_test"] = module
    original_forward = AdaSingle.forward
    try:
        torch.manual_seed(0)
        ada = AdaSingle(dim=8, layers=["attn", "mlp"]).eval()
        cases = [
            (layer, mode, shape)
            for layer in ("attn", "mlp")
            for mode in ("in", "out")
            for shape in [(5, 8), (1, 2, 2, 8)]
        ]
        probes, expected = [], []
        with torch.no_grad():
            for layer, mode, shape in cases:
                hid, emb = torch.randn(*shape), torch.randn(1, 48)
                hid_len = torch.tensor([shape[0]]) if len(shape) == 2 else None
                cache = lambda _key, fn: fn()
                probes.append((hid, emb, layer, mode, hid_len))
                expected.append(ada(hid.clone(), emb, layer=layer, mode=mode, cache=cache, hid_len=hid_len))

        _patch_seedvr_ada_modulation("_seedvr_ada_test")

        with torch.no_grad():
            for (hid, emb, layer, mode, hid_len), exp in zip(probes, expected):
                got = ada(hid.clone(), emb, layer=layer, mode=mode, cache=cache, hid_len=hid_len)
                assert got.shape == exp.shape, (layer, mode)
                assert torch.allclose(exp, got, atol=1e-6), (layer, mode)
    finally:
        AdaSingle.forward = original_forward
        sys.modules.pop("_seedvr_ada_test", None)


def test_seedvr2_video_io_fallback_helpers():
    root = Path(tempfile.mkdtemp())
    try:
        input_path = root / "input.mp4"
        output_path = root / "output.mp4"
        frames = np.zeros((2, 16, 16, 3), dtype=np.uint8)
        frames[1, :, :, 1] = 128
        iio.imwrite(input_path, frames, fps=2)

        video, fps = OVSeedVR2Pipeline.load_video(input_path, max_frames=1)
        assert tuple(video.shape) == (1, 3, 16, 16)
        assert fps is None or fps > 0

        OVSeedVR2Pipeline.save_output(video.permute(1, 0, 2, 3).unsqueeze(0).float() / 127.5 - 1, output_path, fps=2)
        assert output_path.exists()
        assert output_path.stat().st_size > 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_seedvr2_image_preprocess_and_save(tmp_path):
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    write_png(torch.zeros((3, 16, 16), dtype=torch.uint8), str(input_path))

    video, fps = OVSeedVR2Pipeline.load_video(input_path)
    assert fps is None
    preprocessed = OVSeedVR2Pipeline.preprocess_video(video, res_h=16, res_w=16)
    assert tuple(preprocessed.shape) == (1, 3, 1, 16, 16)

    OVSeedVR2Pipeline.save_output(preprocessed, output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0