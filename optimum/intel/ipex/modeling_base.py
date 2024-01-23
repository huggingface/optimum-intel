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
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    GenerationMixin,
    PretrainedConfig,
)
from transformers.utils import WEIGHTS_NAME

from optimum.exporters import TasksManager
from optimum.modeling_base import OptimizedModel

from ..generation.modeling import BaseModelForCausalLM, jit_trace
from ..utils.import_utils import is_torch_version
from ..utils.modeling_utils import patch_decoder_attention_mask


# SUPPORT_MODEL_LIST_FOR_CAUSAL_LM = {"llama": LlamaForCausalLM}
# SUPPORT_TASK_LIST = {"text-generation": SUPPORT_MODEL_LIST_FOR_CAUSAL_LM}

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
        **kwargs,
    ):
        super().__init__(model, config)
        # To do: add XPU support
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self._device)
        self.model_save_dir = model_save_dir

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
        trust_remote_code: bool = False,
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
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
        }

        model = TasksManager.get_model_from_task(task, model_id, **model_kwargs)
        model = patch_decoder_attention_mask(model)
        model = ipex.optimize(model, dtype=torch_dtype, level="O1", auto_kernel_selection=True)
        traced_model = jit_trace(model, task, use_cache)

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)
        torch.jit.save(traced_model, save_dir_path / WEIGHTS_NAME)
        config.torchscript = True

        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            use_auth_token=use_auth_token,
            revision=revision,
            force_download=force_download,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            use_cache=use_cache,
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
        subfolder: str = "",
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
                subfolder=subfolder,
            )
            model_save_dir = Path(model_cache_path).parent

        model = torch.jit.load(model_cache_path)
        torch.jit.freeze(model.eval())
        return cls(model, config=config, model_save_dir=model_save_dir, **kwargs)

    def _save_pretrained(self, save_directory: Union[str, Path]):
        output_path = os.path.join(save_directory, WEIGHTS_NAME)
        torch.jit.save(self.model, output_path)

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


class IPEXModelForSequenceClassification(IPEXModel):
    auto_model_class = AutoModelForSequenceClassification
    export_feature = "text-classification"


class IPEXModelForMaskedLM(IPEXModel):
    auto_model_class = AutoModelForMaskedLM
    export_feature = "fill-mask"


class IPEXModelForTokenClassification(IPEXModel):
    auto_model_class = AutoModelForTokenClassification
    export_feature = "token-classification"


class IPEXModelForCausalLM(IPEXModel, BaseModelForCausalLM):
    auto_model_class = AutoModelForCausalLM
    export_feature = "text-generation"
    forward = BaseModelForCausalLM.forward
    generate = BaseModelForCausalLM.generate
    can_generate = BaseModelForCausalLM.can_generate

    def __init__(
        self,
        model,
        config: PretrainedConfig = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        IPEXModel.__init__(self, model, config)
        BaseModelForCausalLM.__init__(self, model, config, model_save_dir, use_cache, **kwargs)
