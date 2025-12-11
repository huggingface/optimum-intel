"""Compatibility helpers for older Transformers releases."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _register_auto_config(name: str, config_cls) -> None:
    if config_cls is None:
        return

    try:
        from transformers import AutoConfig
        AutoConfig.register(name, config_cls)  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - Transformers API variations
        try:
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING

            if hasattr(CONFIG_MAPPING, "register"):
                CONFIG_MAPPING.register(name, config_cls)  # type: ignore[attr-defined]
            else:
                mapping = getattr(CONFIG_MAPPING, "_extra_content", None)
                if isinstance(mapping, dict):
                    mapping.setdefault(name, config_cls)
        except Exception:
            logger.debug("Failed to register %s with AutoConfig: %s", name, exc)
            return
    else:
        logger.debug("Registered %s with AutoConfig", name)


def ensure_sam2_video_registered() -> None:
    try:
        from transformers import AutoConfig

        if hasattr(AutoConfig, "register"):
            try:
                AutoConfig.get_config("sam2_video")  # type: ignore[attr-defined]
                return
            except Exception:
                pass
    except Exception:
        AutoConfig = None  # type: ignore

    try:
        from transformers.models.sam2_video.configuration_sam2_video import Sam2VideoConfig  # type: ignore
    except Exception:
        Sam2VideoConfig = None

    if AutoConfig is not None:
        try:
            AutoConfig.register("sam2_video", Sam2VideoConfig)  # type: ignore[attr-defined]
            return
        except Exception:
            pass

    _register_auto_config("sam2_video", Sam2VideoConfig)


ensure_sam2_video_registered()

__all__ = ["ensure_sam2_video_registered"]
