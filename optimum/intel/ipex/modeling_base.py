import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Union

import intel_extension_for_pytorch as ipex
import torch
from huggingface_hub import hf_hub_download
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    GenerationMixin,
    PretrainedConfig,
)
from transformers.models.auto.auto_factory import _get_model_class
from transformers.utils import WEIGHTS_NAME

from optimum.exporters import TasksManager
from optimum.modeling_base import OptimizedModel

from ..utils.import_utils import is_torch_version
from ..utils.modeling_utils import patch_decoder_attention_mask
from . import generation_tasks


SUPPORT_MODEL_LIST_FOR_CAUSAL_LM = {
    #  "llama": LlamaForCausalLM
}

SUPPORT_TASK_LIST = {"text-generation": SUPPORT_MODEL_LIST_FOR_CAUSAL_LM}
from ..generation.modeling import jit_trace


logger = logging.getLogger(__name__)


class IPEXModel(OptimizedModel):
    auto_model_class = AutoModel
    export_feature = "feature-extraction"
    base_model_prefix = "ipex_model"

    def __init__(
        self,
        model,
        config: PretrainedConfig = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        OptimizedModel.__init__(self, model=model, config=config)
        # To do: add XPU support
        self._device = torch.device("cpu")
        self.model.to(self._device)

        # Registers the IPEXModelForXXX classes into the transformers AutoModel classes to avoid warnings when creating
        # a pipeline https://github.com/huggingface/transformers/blob/cad61b68396a1a387287a8e2e2fef78a25b79383/src/transformers/pipelines/base.py#L863
        AutoConfig.register(self.base_model_prefix, AutoConfig)
        if hasattr(self.auto_model_class, "register"):
            self.auto_model_class.register(AutoConfig, self.__class__)

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        use_cache: bool = True,
        torch_dtype: Optional[Union[str, "torch.dtype"]] = None,
        **kwargs,
    ):
        if is_torch_version("<", "2.1.0"):
            raise ImportError("`torch>=2.0.0` is needed to trace your model")
        task = cls.export_feature
        model_kwargs = {
            "revision": revision,
            "use_auth_token": use_auth_token,
            "cache_dir": cache_dir,
            "subfolder": subfolder,
            "local_files_only": local_files_only,
            "force_download": force_download,
            "use_cache": use_cache,
            "torch_dtype": torch_dtype,
            "device": "cpu",
        }
        if task not in generation_tasks:
            model_kwargs.pop("use_cache")
        model_type = None
        support_ipex_transformers = False
        if task in SUPPORT_TASK_LIST.keys():
            for name in SUPPORT_TASK_LIST[task].keys():
                if name in model_id:
                    support_ipex_transformers = True
                    model_type = name
                    break

        if support_ipex_transformers and task in SUPPORT_TASK_LIST and model_type in SUPPORT_TASK_LIST[task]:
            # model = SUPPORT_TASK_LIST[task][model_type].from_pretrained(model_id, **model_kwargs)
            pass
        else:
            model = TasksManager.get_model_from_task(task, model_id, **model_kwargs)
            model = patch_decoder_attention_mask(model)

        model = ipex.optimize(model, dtype=torch_dtype, level="O1", auto_kernel_selection=True)

        if kwargs.pop("jit", True):
            try:
                traced_model = cls.apply_jit_optimize(model, task, use_cache, support_ipex_transformers)
                save_dir = TemporaryDirectory()
                save_dir_path = Path(save_dir.name)
                torch.jit.save(traced_model, save_dir_path / WEIGHTS_NAME)
                config.torchscript = True

                return cls._from_pretrained(
                    model_id=save_dir_path,
                    config=config,
                    use_cache=use_cache,
                    use_auth_token=use_auth_token,
                    revision=revision,
                    force_download=force_download,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    model_dtype=torch_dtype,
                    **kwargs,
                )
            except Exception as e:
                logger.warning(f"failed to use PyTorch jit mode due to: {e}.")

        return cls(
            model,
            config=config,
            use_cache=use_cache,
            model_dtype=torch_dtype,
            **kwargs,
        )

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: PretrainedConfig,
        use_auth_token: Optional[Union[bool, str, None]] = None,
        revision: Optional[Union[str, None]] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = WEIGHTS_NAME,
        local_files_only: bool = False,
        use_cache: bool = True,
        **kwargs,
    ):
        # Load the model from local directory
        if os.path.isdir(model_id):
            model_cache_path = os.path.join(model_id, file_name)
            model_save_dir = model_id
        # Download the model from the hub
        else:
            model_cache_path = hf_hub_download(
                repo_id=model_id,
                filename=file_name,
                use_auth_token=use_auth_token,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
            )
            model_save_dir = Path(model_cache_path).parent

        if getattr(config, "torchscript", False):
            model = torch.jit.load(model_cache_path)
            torch.jit.freeze(model.eval())
        else:
            model_class = _get_model_class(config, cls.auto_model_class._model_mapping)
            model = model_class.from_pretrained(model_save_dir)

        return cls(
            model,
            config=config,
            model_save_dir=model_save_dir,
            use_cache=use_cache,
            **kwargs,
        )

    def _save_pretrained(self, save_directory: Union[str, Path], file_name: Optional[str] = None, **kwargs):
        if getattr(self.config, "torchscript", False):
            torch.jit.save(self.model, os.path.join(save_directory, WEIGHTS_NAME))
        else:
            torch.save(self.model, os.path.join(save_directory, WEIGHTS_NAME))

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def eval(self):
        self.model.eval()
        return self

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: Union[torch.device, str]):
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        self.model.to(self._device)
        return self

    def can_generate(self):
        return isinstance(self.model, GenerationMixin)

    def generate(self, *args, **kwargs):
        if not self.can_generate():
            raise TypeError(
                f"The current model class {self.model.__class__} is not compatible with `.generate()`, as it doesn't have a language model head."
            )
        return self.model.generate(*args, **kwargs)

    @classmethod
    def apply_jit_optimize(cls, model, task, use_cache, support_ipex_transformers=False):
        return jit_trace(model, task, use_cache)


class IPEXModelForSequenceClassification(IPEXModel):
    auto_model_class = AutoModelForSequenceClassification
    export_feature = "text-classification"
