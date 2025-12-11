"""SAM2 compatibility hooks for Optimum OpenVINO exporters."""

from __future__ import annotations

import transformers

try:  # new Transformers no longer expose MT5Tokenizer
    from transformers import MT5Tokenizer  # type: ignore[attr-defined]
except ImportError:  # transformers >= version dropping MT5Tokenizer
    from transformers import T5Tokenizer

    class MT5Tokenizer(T5Tokenizer):  # type: ignore[misc]
        pass

    setattr(transformers, "MT5Tokenizer", MT5Tokenizer)

_SAM2_ERROR_TOKEN = "positional_embedding"


def _patch_sam2_config():
    try:
        from transformers.models.sam2.configuration_sam2 import Sam2Config  # type: ignore
    except Exception:
        Sam2Config = None

    try:
        from transformers.models.sam2_video.configuration_sam2_video import Sam2VideoConfig  # type: ignore
    except Exception:
        Sam2VideoConfig = None

    def _guard(cfg_cls):
        if cfg_cls is None or getattr(cfg_cls, "_optimum_config_patched", False):
            return
        original_init = cfg_cls.__init__

        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            try:
                if getattr(self, "tie_word_embeddings", True):
                    self.tie_word_embeddings = False
            except Exception:
                pass

            try:
                model_type = getattr(self, "model_type", None)
                if model_type == "sam2_video":
                    mapping = dict(getattr(self, "export_model_type_map", {}) or {})
                    mapping.setdefault("feature-extraction", "sam2video_vision_encoder")
                    mapping.setdefault("image-segmentation", "sam2video_mask_decoder")
                    self.export_model_type_map = mapping
                    if getattr(self, "export_model_type", None) is None:
                        self.export_model_type = mapping.get("feature-extraction")
            except Exception:
                pass

        cfg_cls.__init__ = patched_init
        setattr(cfg_cls, "_optimum_config_patched", True)

    _guard(Sam2Config)
    _guard(Sam2VideoConfig)


def _patch_sam2_mark_tied_weights():
    try:
        from transformers.models.sam2.modeling_sam2 import Sam2Model  # type: ignore
    except Exception:  # transformers may not ship sam2 yet
        Sam2Model = None

    try:
        from transformers.models.sam2_video.modeling_sam2_video import Sam2VideoModel  # type: ignore
    except Exception:
        Sam2VideoModel = None

    def _guard(model_cls):
        if model_cls is None:
            return
        original = getattr(model_cls, "mark_tied_weights_as_initialized", None)
        if original is None or getattr(model_cls, "_optimum_mark_tied_weights_patched", False):
            return

        def patched(self, *args, **kwargs):
            tied = getattr(self, "_tied_weights_keys", None)
            if tied and not getattr(self, "_optimum_sam2_ties_filtered", False):
                filtered = []
                removed = False
                for pair in tied:
                    keys = pair if isinstance(pair, (list, tuple, set)) else (pair,)
                    if any((_SAM2_ERROR_TOKEN in str(key)) for key in keys if key):
                        removed = True
                        continue
                    filtered.append(pair)
                if removed:
                    try:
                        self._tied_weights_keys = type(tied)(filtered)
                    except Exception:
                        self._tied_weights_keys = filtered
                setattr(self, "_optimum_sam2_ties_filtered", True)
            config = getattr(self, "config", None)
            if config is not None and getattr(config, "tie_word_embeddings", None):
                try:
                    config.tie_word_embeddings = False
                except Exception:
                    pass
            try:
                return original(self, *args, **kwargs)
            except AttributeError as err:
                if _SAM2_ERROR_TOKEN in str(err):
                    # Tied metadata can sporadically include buffers; skip them quietly.
                    return
                raise

        model_cls.mark_tied_weights_as_initialized = patched
        setattr(model_cls, "_optimum_mark_tied_weights_patched", True)

    _guard(Sam2Model)
    _guard(Sam2VideoModel)


_patch_sam2_config()
_patch_sam2_mark_tied_weights()

__all__ = []
