#  Copyright 2022 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import inspect
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from itertools import chain

import datasets
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, RandomSampler
from torch.onnx import export as onnx_export

from transformers import PreTrainedModel, default_data_collator
from huggingface_hub import HfApi, hf_hub_download
from transformers.onnx import FeaturesManager, export
from transformers.onnx.utils import get_preprocessor

from nncf import NNCFConfig
from nncf.torch import create_compressed_model, register_default_init_args
from nncf.torch.initialization import PTInitializingDataLoader
from nncf.torch.dynamic_graph.io_handling import wrap_nncf_model_inputs_with_objwalk
from optimum.quantization_base import OptimumQuantizer

from .utils import ONNX_WEIGHTS_NAME


class OVDataLoader(PTInitializingDataLoader):
    def get_inputs(self, dataloader_output) -> Tuple[Tuple, Dict]:
        return (), dataloader_output


class OVQuantizer(OptimumQuantizer):
    """
    Handles the NNCF quantization process.
    """

    def __init__(self, model: PreTrainedModel, **kwargs):
        """
        Args:
            model (`PreTrainedModel`):
                The model to quantize.
        """
        super().__init__()
        self.model = model
        self.seed = kwargs.pop("seed", 42)
        signature = inspect.signature(self.model.forward)
        self._signature_columns = list(signature.parameters.keys())
        self.input_names = None

    @classmethod
    def from_pretrained(cls, model: PreTrainedModel, **kwargs):
        # TODO : Create model
        return cls(model, **kwargs)

    def quantize(
        self,
        quantization_config: NNCFConfig,
        calibration_dataset: Dataset,
        save_directory: Union[str, Path],
        file_name: Optional[str] = None,
        batch_size: int = 8,
    ):

        '''print(dir(self.model))
        print(self.model.__class__.__name__)
        print(dir(type(self.model).__class__))
        return'''

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        file_name = file_name if file_name is not None else ONNX_WEIGHTS_NAME
        output_path = save_directory.joinpath(file_name)
        calibration_dataloader = self._get_calibration_dataloader(calibration_dataset, batch_size)
        quantization_config = register_default_init_args(quantization_config, calibration_dataloader)
        controller, compressed_model = create_compressed_model(self.model, quantization_config, wrap_inputs_fn=wrap_nncf_model_inputs_with_objwalk)
        controller.prepare_for_export()

        
        model_id = 'distilbert-base-uncased-finetuned-sst-2-english'
        task = "sequence-classification"

        # TODO: support private models
        preprocessor = get_preprocessor(model_id)
        # TODO: Add framework ["pt", "tf"]
        model = FeaturesManager.get_model_from_feature(task, model_id)
        _, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=task)
        onnx_config = model_onnx_config(model.config)

        print("Export to ONNX")

        # Export the model to the ONNX format
        '''export(
            preprocessor=preprocessor,
            model=compressed_model,
            config=onnx_config,
            opset=onnx_config.default_onnx_opset,
            output=output_path,
        )'''

        
        model_inputs = onnx_config.generate_dummy_inputs(preprocessor, framework="pt")
        model.config.return_dict = True
        onnx_config.patch_ops()

        compressed_model.config.return_dict = True
        compressed_model.eval()
        

        print(model_inputs)

        print(self.input_names)

        print(list(onnx_config.inputs.keys()))

        loaded_item = next(iter(calibration_dataloader))
        _model_inputs = calibration_dataloader.get_inputs(loaded_item)
        print(_model_inputs)

        '''original_forward = compressed_model.forward
        args = self._model_args[:-1]
        kwargs = self._model_args[-1]
        model.forward = partial(model.forward, *args, **kwargs)'''

        # pylint:disable=unexpected-keyword-arg
        with torch.no_grad():
            # Should call this, otherwise the operations executed during export will end up in the graph.
            compressed_model.disable_dynamic_graph_building()        

            '''onnx_export(
                        model,
                        tuple(model_inputs,),
                        f=output_path.as_posix(),
                        input_names=list(onnx_config.inputs.keys()),
                        output_names=list(onnx_config.outputs.keys()),
                        dynamic_axes={name: axes for name, axes in chain(onnx_config.inputs.items(), onnx_config.outputs.items())},
                        opset_version=13,
                        training=True
                    )'''
            dummy_inputs = list(loaded_item.values())
            

            onnx_export(
                    compressed_model,
                    tuple(dummy_inputs),
                    f=output_path.as_posix(),
                    input_names=list(onnx_config.inputs.keys()),
                    output_names=list(onnx_config.outputs.keys()),
                    dynamic_axes={name: axes for name, axes in chain(onnx_config.inputs.items(), onnx_config.outputs.items())},
                    do_constant_folding=True,
                    opset_version=13,
                )

            compressed_model.enable_dynamic_graph_building()
        #compressed_model.forward = original_forward

        print("Finished export")

        #controller.export_model(output_path.as_posix(), input_names=list(onnx_config.inputs.keys()))
        return output_path

    def get_calibration_dataset(
        self,
        dataset_name: str,
        num_samples: int = 100,
        dataset_config_name: Optional[str] = None,
        dataset_split: Optional[str] = None,
        preprocess_function: Optional[Callable] = None,
        preprocess_batch: bool = True,
        use_auth_token: bool = False,
    ) -> Dataset:
        calib_dataset = load_dataset(
            dataset_name,
            name=dataset_config_name,
            split=dataset_split,
            use_auth_token=use_auth_token,
        )

        if num_samples is not None:
            num_samples = min(num_samples, len(calib_dataset))
            calib_dataset = calib_dataset.shuffle(seed=self.seed).select(range(num_samples))

        if preprocess_function is not None:
            calib_dataset = calib_dataset.map(preprocess_function, batched=preprocess_batch)

        return calib_dataset

    def _get_calibration_dataloader(self, calibration_dataset: Dataset, batch_size: int) -> OVDataLoader:
        calibration_dataset = self._remove_unused_columns(calibration_dataset)
        self.input_names = calibration_dataset.column_names
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        sampler = RandomSampler(calibration_dataset, generator=generator)
        calibration_dataloader = DataLoader(
            calibration_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=default_data_collator,
            drop_last=False,
        )
        return OVDataLoader(calibration_dataloader)

    def _remove_unused_columns(self, dataset: Dataset):
        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
        return dataset.remove_columns(ignored_columns)
