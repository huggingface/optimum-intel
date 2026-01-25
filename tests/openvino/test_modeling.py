#  Copyright 2021 The HuggingFace Team. All rights reserved.
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

import gc
import os
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Generator

import numpy as np
import open_clip  # type: ignore
import openvino as ov
import pytest
import requests
import timm
import torch
from datasets import load_dataset
from evaluate import evaluator
from huggingface_hub import snapshot_download
from parameterized import parameterized
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForAudioClassification,
    AutoModelForAudioFrameClassification,
    AutoModelForAudioXVector,
    AutoModelForCausalLM,
    AutoModelForCTC,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForZeroShotImageClassification,
    AutoProcessor,
    AutoTokenizer,
    PretrainedConfig,
    SamModel,
    pipeline,
    set_seed,
)
from transformers.onnx.utils import get_preprocessor
from transformers.testing_utils import slow
from transformers.utils import http_user_agent
from utils_tests import F32_CONFIG, MODEL_NAMES, OPENVINO_DEVICE, SEED, TENSOR_ALIAS_TO_TYPE, TEST_IMAGE_URL

from optimum.intel import (
    OVDiffusionPipeline,
    OVFluxPipeline,
    OVModelForAudioClassification,
    OVModelForAudioFrameClassification,
    OVModelForAudioXVector,
    OVModelForCausalLM,
    OVModelForCTC,
    OVModelForCustomTasks,
    OVModelForFeatureExtraction,
    OVModelForImageClassification,
    OVModelForMaskedLM,
    OVModelForQuestionAnswering,
    OVModelForSeq2SeqLM,
    OVModelForSequenceClassification,
    OVModelForTextToSpeechSeq2Seq,
    OVModelForTokenClassification,
    OVModelForVisualCausalLM,
    OVModelForZeroShotImageClassification,
    OVModelOpenCLIPForZeroShotImageClassification,
    OVSamModel,
    OVSentenceTransformer,
    OVStableDiffusionPipeline,
)
from optimum.intel.openvino import OV_DECODER_NAME, OV_DECODER_WITH_PAST_NAME, OV_ENCODER_NAME, OV_XML_FILE_NAME
from optimum.intel.openvino.modeling_base import OVBaseModel
from optimum.intel.openvino.modeling_timm import TimmImageProcessor
from optimum.intel.openvino.modeling_visual_language import MODEL_PARTS_CLS_MAPPING, MODEL_TYPE_TO_CLS_MAPPING
from optimum.intel.openvino.utils import (
    OV_LANGUAGE_MODEL_NAME,
    OV_PROMPT_ENCODER_MASK_DECODER_MODEL_NAME,
    OV_TEXT_EMBEDDINGS_MODEL_NAME,
    OV_VISION_EMBEDDINGS_MODEL_NAME,
    OV_VISION_ENCODER_MODEL_NAME,
    TemporaryDirectory,
)
from optimum.intel.pipelines import pipeline as optimum_pipeline
from optimum.intel.utils.import_utils import _langchain_hf_available, is_transformers_version
from optimum.intel.utils.modeling_utils import _find_files_matching_pattern
from optimum.utils import (
    DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER,
    DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
    DIFFUSION_MODEL_TRANSFORMER_SUBFOLDER,
    DIFFUSION_MODEL_UNET_SUBFOLDER,
    DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
    DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
)
from optimum.utils.testing_utils import require_diffusers


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class OVModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.OV_MODEL_ID = "echarlaix/distilbert-base-uncased-finetuned-sst-2-english-openvino"
        self.OV_DECODER_MODEL_ID = "helenai/gpt2-ov"
        self.OV_SEQ2SEQ_MODEL_ID = "echarlaix/t5-small-openvino"
        self.OV_SD_DIFFUSION_MODEL_ID = "katuni4ka/tiny-stable-diffusion-openvino"
        self.OV_FLUX_DIFFUSION_MODEL_ID = "katuni4ka/tiny-random-flux-ov"
        self.OV_VLM_MODEL_ID = "katuni4ka/tiny-random-llava-ov"
        self.OV_SAM_MODEL_ID = "katuni4ka/sam-vit-tiny-random-ov"
        self.OV_TEXTSPEECH_MODEL_ID = "optimum-internal-testing/tiny-random-SpeechT5ForTextToSpeech-openvino"

    def test_load_from_hub_and_save_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.OV_MODEL_ID)
        tokens = tokenizer("This is a sample input", return_tensors="pt")
        loaded_model = OVModelForSequenceClassification.from_pretrained(self.OV_MODEL_ID, device=OPENVINO_DEVICE)
        self.assertIsInstance(loaded_model.config, PretrainedConfig)
        # Test that PERFORMANCE_HINT is set to LATENCY by default
        self.assertEqual(loaded_model.ov_config.get("PERFORMANCE_HINT"), "LATENCY")
        self.assertEqual(loaded_model.request.get_property("PERFORMANCE_HINT"), "LATENCY")
        loaded_model_outputs = loaded_model(**tokens)

        # Test specifying ov_config with throughput hint and manual cache dir
        manual_openvino_cache_dir = loaded_model.model_save_dir / "manual_model_cache"
        ov_config = {"CACHE_DIR": str(manual_openvino_cache_dir), "PERFORMANCE_HINT": "THROUGHPUT"}
        loaded_model = OVModelForSequenceClassification.from_pretrained(
            self.OV_MODEL_ID, ov_config=ov_config, device=OPENVINO_DEVICE
        )
        self.assertTrue(manual_openvino_cache_dir.is_dir())
        num_blobs = len(list(manual_openvino_cache_dir.glob("*.blob")))
        self.assertGreaterEqual(num_blobs, 1)
        self.assertEqual(loaded_model.request.get_property("PERFORMANCE_HINT"), "THROUGHPUT")

        # Test compile only

        compile_only_model = OVModelForSequenceClassification.from_pretrained(
            self.OV_MODEL_ID, ov_config=ov_config, compile_only=True, device=OPENVINO_DEVICE
        )
        self.assertTrue(manual_openvino_cache_dir.is_dir())
        current_num_blobs = len(list(manual_openvino_cache_dir.glob("*.blob")))
        # compile_only get model from cache
        self.assertGreaterEqual(current_num_blobs, num_blobs)
        self.assertIsInstance(compile_only_model.model, ov.CompiledModel)
        self.assertIsInstance(compile_only_model.request, ov.CompiledModel)
        outputs = compile_only_model(**tokens)
        self.assertTrue(torch.equal(loaded_model_outputs.logits, outputs.logits))
        del compile_only_model

        with TemporaryDirectory() as tmpdirname:
            loaded_model.save_pretrained(tmpdirname)
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(OV_XML_FILE_NAME in folder_contents)
            self.assertTrue(OV_XML_FILE_NAME.replace(".xml", ".bin") in folder_contents)
            model = OVModelForSequenceClassification.from_pretrained(
                tmpdirname, ov_config={"NUM_STREAMS": 2}, device=OPENVINO_DEVICE
            )
            # Test that PERFORMANCE_HINT is set to LATENCY by default even with ov_config provided
            self.assertEqual(model.ov_config.get("PERFORMANCE_HINT"), "LATENCY")
            self.assertEqual(model.request.get_property("PERFORMANCE_HINT"), "LATENCY")

        outputs = model(**tokens)
        self.assertTrue(torch.equal(loaded_model_outputs.logits, outputs.logits))

        del loaded_model
        del model
        gc.collect()

    @parameterized.expand((True, False))
    def test_load_from_hub_and_save_decoder_model(self, use_cache):
        model_id = MODEL_NAMES["gpt2-with-cache-ov"] if use_cache else MODEL_NAMES["gpt2-without-cache-ov"]

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer("This is a sample input", return_tensors="pt")
        loaded_model = OVModelForCausalLM.from_pretrained(model_id, use_cache=use_cache, device=OPENVINO_DEVICE)
        self.assertIsInstance(loaded_model.config, PretrainedConfig)
        # Test that PERFORMANCE_HINT is set to LATENCY by default
        self.assertEqual(loaded_model.ov_config.get("PERFORMANCE_HINT"), "LATENCY")
        self.assertEqual(loaded_model.request.get_compiled_model().get_property("PERFORMANCE_HINT"), "LATENCY")
        loaded_model_outputs = loaded_model(**tokens)

        with TemporaryDirectory() as tmpdirname:
            loaded_model.save_pretrained(tmpdirname)
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(OV_XML_FILE_NAME in folder_contents)
            self.assertTrue(OV_XML_FILE_NAME.replace(".xml", ".bin") in folder_contents)
            model = OVModelForCausalLM.from_pretrained(tmpdirname, use_cache=use_cache, device=OPENVINO_DEVICE)
            self.assertEqual(model.use_cache, use_cache)

            compile_only_model = OVModelForCausalLM.from_pretrained(
                tmpdirname, compile_only=True, use_cache=use_cache, device=OPENVINO_DEVICE
            )
            self.assertIsInstance(compile_only_model.model, ov.CompiledModel)
            self.assertIsInstance(compile_only_model.request, ov.InferRequest)
            outputs = compile_only_model(**tokens)
            self.assertTrue(torch.equal(loaded_model_outputs.logits, outputs.logits))
            del compile_only_model

        outputs = model(**tokens)
        self.assertTrue(torch.equal(loaded_model_outputs.logits, outputs.logits))
        del loaded_model
        del model
        gc.collect()

    def test_load_from_hub_and_save_visual_language_model(self):
        model_ids = [self.OV_VLM_MODEL_ID]
        if is_transformers_version(">=", "4.51") and is_transformers_version("<", "4.57"):
            # the phi4 auto-processor can't be loaded in offline mode
            # anymore due to an internal bug in transformers
            model_ids.append("katuni4ka/phi-4-multimodal-ov")
        for model_id in model_ids:
            processor = get_preprocessor(model_id)
            prompt = "What is shown in this image?"
            image = Image.open(
                requests.get(
                    TEST_IMAGE_URL,
                    stream=True,
                ).raw
            )
            loaded_model = OVModelForVisualCausalLM.from_pretrained(model_id, device=OPENVINO_DEVICE)
            self.assertIsInstance(loaded_model, MODEL_TYPE_TO_CLS_MAPPING[loaded_model.config.model_type])
            for component_name, component in loaded_model.components.items():
                self.assertIsInstance(component, MODEL_PARTS_CLS_MAPPING[component_name])
            self.assertIsInstance(loaded_model.config, PretrainedConfig)
            # Test that PERFORMANCE_HINT is set to LATENCY by default
            self.assertEqual(loaded_model.ov_config.get("PERFORMANCE_HINT"), "LATENCY")

            for component_name, component in loaded_model.components.items():
                self.assertIsInstance(component.model, ov.Model)
                if component_name == "language_model":
                    self.assertEqual(
                        component.request.get_compiled_model().get_property("PERFORMANCE_HINT"), "LATENCY"
                    )
                    self.assertIsInstance(component.text_emb_model, ov.Model)
                    self.assertEqual(component.text_emb_request.get_property("PERFORMANCE_HINT"), "LATENCY")
                else:
                    self.assertEqual(component.request.get_property("PERFORMANCE_HINT"), "LATENCY")
            if "llava" in model_id:
                processor.patch_size = loaded_model.config.vision_config.patch_size
            inputs = loaded_model.preprocess_inputs(text=prompt, image=image, processor=processor)
            set_seed(SEED)
            loaded_model_outputs = loaded_model(**inputs)

            with TemporaryDirectory() as tmpdirname:
                loaded_model.save_pretrained(tmpdirname)
                folder_contents = os.listdir(tmpdirname)
                model_files = [
                    OV_LANGUAGE_MODEL_NAME,
                    OV_TEXT_EMBEDDINGS_MODEL_NAME,
                    OV_VISION_EMBEDDINGS_MODEL_NAME,
                ]
                model_files += [f"openvino_{part}_model.xml" for part in loaded_model.additional_parts]
                for xml_file_name in model_files:
                    self.assertTrue(xml_file_name in folder_contents)
                    self.assertTrue(xml_file_name.replace(".xml", ".bin") in folder_contents)
                model = OVModelForVisualCausalLM.from_pretrained(tmpdirname, device=OPENVINO_DEVICE)
                compile_only_model = OVModelForVisualCausalLM.from_pretrained(
                    tmpdirname, compile_only=True, device=OPENVINO_DEVICE
                )
                for ov_model in compile_only_model.ov_models.values():
                    self.assertIsInstance(ov_model, ov.CompiledModel)
                for component_name, component in compile_only_model.components.items():
                    self.assertIsInstance(component.model, ov.CompiledModel)
                    if component_name == "language_model":
                        self.assertIsInstance(component.request, ov.InferRequest)
                        self.assertIsInstance(component.text_emb_model, ov.CompiledModel)
                        self.assertIsInstance(component.text_emb_request, ov.CompiledModel)
                    else:
                        self.assertIsInstance(component.request, ov.CompiledModel)

                outputs = compile_only_model(**inputs)
                self.assertTrue(torch.equal(loaded_model_outputs.logits, outputs.logits))
                del compile_only_model

            outputs = model(**inputs)
            self.assertTrue(torch.equal(loaded_model_outputs.logits, outputs.logits))
            del loaded_model
            del model
            gc.collect()

    def test_load_from_hub_and_save_seq2seq_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.OV_SEQ2SEQ_MODEL_ID)
        tokens = tokenizer("This is a sample input", return_tensors="pt")
        loaded_model = OVModelForSeq2SeqLM.from_pretrained(
            self.OV_SEQ2SEQ_MODEL_ID, compile=False, device=OPENVINO_DEVICE
        )
        self.assertIsInstance(loaded_model.config, PretrainedConfig)
        loaded_model.to("cpu")
        loaded_model.compile()
        # Test that PERFORMANCE_HINT is set to LATENCY by default
        self.assertEqual(loaded_model.ov_config.get("PERFORMANCE_HINT"), "LATENCY")
        self.assertEqual(loaded_model.decoder.request.get_compiled_model().get_property("PERFORMANCE_HINT"), "LATENCY")

        loaded_model_outputs = loaded_model.generate(**tokens)

        with TemporaryDirectory() as tmpdirname:
            loaded_model.save_pretrained(tmpdirname)
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(OV_ENCODER_NAME in folder_contents)
            self.assertTrue(OV_DECODER_NAME in folder_contents)
            self.assertTrue(OV_DECODER_WITH_PAST_NAME in folder_contents)
            model = OVModelForSeq2SeqLM.from_pretrained(tmpdirname, device="cpu")
            # compile only
            compile_only_model = OVModelForSeq2SeqLM.from_pretrained(
                tmpdirname, compile_only=True, device=OPENVINO_DEVICE
            )
            self.assertIsInstance(compile_only_model.encoder.model, ov.CompiledModel)
            self.assertIsInstance(compile_only_model.decoder.model, ov.CompiledModel)
            self.assertIsInstance(compile_only_model.decoder_with_past.model, ov.CompiledModel)
            outputs = compile_only_model.generate(**tokens)
            self.assertTrue(torch.equal(loaded_model_outputs, outputs))
            del compile_only_model

        outputs = model.generate(**tokens)
        self.assertTrue(torch.equal(loaded_model_outputs, outputs))
        del loaded_model
        del model
        gc.collect()

    @require_diffusers
    def test_load_from_hub_and_save_stable_diffusion_model(self):
        loaded_pipeline = OVStableDiffusionPipeline.from_pretrained(
            self.OV_SD_DIFFUSION_MODEL_ID, compile=False, device=OPENVINO_DEVICE
        )
        self.assertIsInstance(loaded_pipeline.config, Dict)
        # Test that PERFORMANCE_HINT is set to LATENCY by default
        self.assertEqual(loaded_pipeline.ov_config.get("PERFORMANCE_HINT"), "LATENCY")
        loaded_pipeline.compile()
        self.assertEqual(loaded_pipeline.unet.request.get_property("PERFORMANCE_HINT"), "LATENCY")
        batch_size, height, width = 2, 16, 16
        inputs = {
            "prompt": ["sailing ship in storm by Leonardo da Vinci"] * batch_size,
            "height": height,
            "width": width,
            "num_inference_steps": 2,
            "output_type": "np",
        }

        np.random.seed(0)
        torch.manual_seed(0)
        pipeline_outputs = loaded_pipeline(**inputs).images
        self.assertEqual(pipeline_outputs.shape, (batch_size, height, width, 3))

        with TemporaryDirectory() as tmpdirname:
            loaded_pipeline.save_pretrained(tmpdirname)
            pipeline = OVStableDiffusionPipeline.from_pretrained(tmpdirname, device=OPENVINO_DEVICE)
            folder_contents = os.listdir(tmpdirname)
            self.assertIn(loaded_pipeline.config_name, folder_contents)
            for subfoler in {
                DIFFUSION_MODEL_UNET_SUBFOLDER,
                DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
                DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
                DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
            }:
                folder_contents = os.listdir(os.path.join(tmpdirname, subfoler))
                self.assertIn(OV_XML_FILE_NAME, folder_contents)
                self.assertIn(OV_XML_FILE_NAME.replace(".xml", ".bin"), folder_contents)

            compile_only_pipeline = OVStableDiffusionPipeline.from_pretrained(
                tmpdirname, compile_only=True, device=OPENVINO_DEVICE
            )
            self.assertIsInstance(compile_only_pipeline.unet.model, ov.CompiledModel)
            self.assertIsInstance(compile_only_pipeline.text_encoder.model, ov.CompiledModel)
            self.assertIsInstance(compile_only_pipeline.vae_encoder.model, ov.CompiledModel)
            self.assertIsInstance(compile_only_pipeline.vae_decoder.model, ov.CompiledModel)

            np.random.seed(0)
            torch.manual_seed(0)
            outputs = compile_only_pipeline(**inputs).images
            np.testing.assert_allclose(pipeline_outputs, outputs, atol=1e-4, rtol=1e-4)
            del compile_only_pipeline

        np.random.seed(0)
        torch.manual_seed(0)
        outputs = pipeline(**inputs).images
        np.testing.assert_allclose(pipeline_outputs, outputs, atol=1e-4, rtol=1e-4)
        del pipeline
        gc.collect()

    @require_diffusers
    def test_load_from_hub_and_save_flux_model(self):
        loaded_pipeline = OVDiffusionPipeline.from_pretrained(
            self.OV_FLUX_DIFFUSION_MODEL_ID, compile=False, device=OPENVINO_DEVICE
        )
        self.assertIsInstance(loaded_pipeline, OVFluxPipeline)
        self.assertIsInstance(loaded_pipeline.config, Dict)
        # Test that PERFORMANCE_HINT is set to LATENCY by default
        self.assertEqual(loaded_pipeline.ov_config.get("PERFORMANCE_HINT"), "LATENCY")
        loaded_pipeline.compile()
        self.assertIsNone(loaded_pipeline.unet)
        self.assertEqual(loaded_pipeline.transformer.request.get_property("PERFORMANCE_HINT"), "LATENCY")
        batch_size, height, width = 2, 16, 16
        inputs = {
            "prompt": ["sailing ship in storm by Leonardo da Vinci"] * batch_size,
            "height": height,
            "width": width,
            "num_inference_steps": 2,
            "output_type": "np",
        }

        np.random.seed(0)
        torch.manual_seed(0)
        pipeline_outputs = loaded_pipeline(**inputs).images
        self.assertEqual(pipeline_outputs.shape, (batch_size, height, width, 3))

        with TemporaryDirectory() as tmpdirname:
            loaded_pipeline.save_pretrained(tmpdirname)
            pipeline = OVDiffusionPipeline.from_pretrained(tmpdirname, device=OPENVINO_DEVICE)
            self.assertIsInstance(loaded_pipeline, OVFluxPipeline)
            folder_contents = os.listdir(tmpdirname)
            self.assertIn(loaded_pipeline.config_name, folder_contents)
            for subfoler in {
                DIFFUSION_MODEL_TRANSFORMER_SUBFOLDER,
                DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
                DIFFUSION_MODEL_TEXT_ENCODER_2_SUBFOLDER,
                DIFFUSION_MODEL_VAE_ENCODER_SUBFOLDER,
                DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
            }:
                folder_contents = os.listdir(os.path.join(tmpdirname, subfoler))
                self.assertIn(OV_XML_FILE_NAME, folder_contents)
                self.assertIn(OV_XML_FILE_NAME.replace(".xml", ".bin"), folder_contents)

            compile_only_pipeline = OVDiffusionPipeline.from_pretrained(
                tmpdirname, compile_only=True, device=OPENVINO_DEVICE
            )
            self.assertIsInstance(compile_only_pipeline, OVFluxPipeline)
            self.assertIsInstance(compile_only_pipeline.transformer.model, ov.CompiledModel)
            self.assertIsInstance(compile_only_pipeline.text_encoder.model, ov.CompiledModel)
            self.assertIsInstance(compile_only_pipeline.text_encoder_2.model, ov.CompiledModel)
            self.assertIsInstance(compile_only_pipeline.vae_encoder.model, ov.CompiledModel)
            self.assertIsInstance(compile_only_pipeline.vae_decoder.model, ov.CompiledModel)

            np.random.seed(0)
            torch.manual_seed(0)
            outputs = compile_only_pipeline(**inputs).images
            np.testing.assert_allclose(pipeline_outputs, outputs, atol=1e-4, rtol=1e-4)
            del compile_only_pipeline

        np.random.seed(0)
        torch.manual_seed(0)
        outputs = pipeline(**inputs).images
        np.testing.assert_allclose(pipeline_outputs, outputs, atol=1e-4, rtol=1e-4)
        del pipeline
        gc.collect()

    def test_load_from_hub_and_save_sam_model(self):
        loaded_model = OVModelForFeatureExtraction.from_pretrained(self.OV_SAM_MODEL_ID, device=OPENVINO_DEVICE)
        self.assertIsInstance(loaded_model, OVSamModel)
        self.assertIsInstance(loaded_model.config, PretrainedConfig)
        # Test that PERFORMANCE_HINT is not set by default
        self.assertIsNone(loaded_model.ov_config.get("PERFORMANCE_HINT"))

        # Test specifying ov_config with throughput hint and manual cache dir
        manual_openvino_cache_dir = loaded_model.model_save_dir / "manual_model_cache"
        ov_config = {"CACHE_DIR": str(manual_openvino_cache_dir), "PERFORMANCE_HINT": "THROUGHPUT"}
        loaded_model = OVModelForFeatureExtraction.from_pretrained(
            self.OV_SAM_MODEL_ID, ov_config=ov_config, device=OPENVINO_DEVICE
        )

        self.assertTrue(manual_openvino_cache_dir.is_dir())
        num_blobs = len(list(manual_openvino_cache_dir.glob("*.blob")))
        self.assertGreaterEqual(num_blobs, 2)
        self.assertEqual(loaded_model.vision_encoder.request.get_property("PERFORMANCE_HINT"), "THROUGHPUT")
        self.assertEqual(
            loaded_model.prompt_encoder_mask_decoder.request.get_property("PERFORMANCE_HINT"), "THROUGHPUT"
        )
        processor = get_preprocessor(self.OV_SAM_MODEL_ID)
        img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
        input_points = [[[450, 600]]]
        raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
        inputs = processor(raw_image, input_points=input_points, return_tensors="pt")
        loaded_model_outputs = loaded_model(**inputs)

        # Test compile only
        compile_only_model = OVModelForFeatureExtraction.from_pretrained(
            self.OV_SAM_MODEL_ID, ov_config=ov_config, compile_only=True, device=OPENVINO_DEVICE
        )
        self.assertTrue(manual_openvino_cache_dir.is_dir())
        current_num_blobs = len(list(manual_openvino_cache_dir.glob("*.blob")))
        # compile_only get model from cache
        self.assertGreaterEqual(current_num_blobs, num_blobs)
        self.assertIsInstance(compile_only_model.vision_encoder.model, ov.CompiledModel)
        self.assertIsInstance(compile_only_model.vision_encoder.request, ov.CompiledModel)
        self.assertIsInstance(compile_only_model.prompt_encoder_mask_decoder.model, ov.CompiledModel)
        self.assertIsInstance(compile_only_model.prompt_encoder_mask_decoder.request, ov.CompiledModel)
        outputs = compile_only_model(**inputs)
        torch.testing.assert_close(loaded_model_outputs.iou_scores, outputs.iou_scores, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(loaded_model_outputs.pred_masks, outputs.pred_masks, atol=1e-4, rtol=1e-4)

        # Test save and load
        with TemporaryDirectory() as tmpdirname:
            loaded_model.save_pretrained(tmpdirname)
            folder_contents = os.listdir(tmpdirname)
            for ir_file in [OV_VISION_ENCODER_MODEL_NAME, OV_PROMPT_ENCODER_MASK_DECODER_MODEL_NAME]:
                self.assertTrue(ir_file in folder_contents)
                self.assertTrue(ir_file.replace(".xml", ".bin") in folder_contents)
            model = OVModelForFeatureExtraction.from_pretrained(
                tmpdirname, ov_config={"NUM_STREAMS": 2}, device=OPENVINO_DEVICE
            )
            self.assertEqual(loaded_model.vision_encoder.request.get_property("PERFORMANCE_HINT"), "THROUGHPUT")
            self.assertEqual(
                loaded_model.prompt_encoder_mask_decoder.request.get_property("PERFORMANCE_HINT"), "THROUGHPUT"
            )

        outputs = model(**inputs)
        torch.testing.assert_close(loaded_model_outputs.iou_scores, outputs.iou_scores, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(loaded_model_outputs.pred_masks, outputs.pred_masks, atol=1e-4, rtol=1e-4)

    def test_load_from_hub_and_save_text_speech_model(self):
        loaded_model = OVModelForTextToSpeechSeq2Seq.from_pretrained(
            self.OV_TEXTSPEECH_MODEL_ID, device=OPENVINO_DEVICE
        )
        self.assertIsInstance(loaded_model.config, PretrainedConfig)
        # Test that PERFORMANCE_HINT is set to LATENCY by default
        self.assertEqual(loaded_model.ov_config.get("PERFORMANCE_HINT"), "LATENCY")

        processor = AutoProcessor.from_pretrained(self.OV_TEXTSPEECH_MODEL_ID)
        text_data = "This text is converted to speech using OpenVINO backend"
        inputs = processor(text=text_data, return_tensors="pt")
        speaker_embeddings = np.random.randn(1, 512).astype(np.float32)
        loaded_model_outputs = loaded_model.generate(
            input_ids=inputs["input_ids"], speaker_embeddings=speaker_embeddings
        )

        with TemporaryDirectory() as tmpdirname:
            loaded_model.save_pretrained(tmpdirname)
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(loaded_model._ov_model_paths["encoder"] in folder_contents)
            self.assertTrue(loaded_model._ov_model_paths["decoder"] in folder_contents)
            self.assertTrue(loaded_model._ov_model_paths["postnet"] in folder_contents)
            self.assertTrue(loaded_model._ov_model_paths["vocoder"] in folder_contents)
            model = OVModelForTextToSpeechSeq2Seq.from_pretrained(tmpdirname, device="cpu")
            # compile only
            compile_only_model = OVModelForTextToSpeechSeq2Seq.from_pretrained(
                tmpdirname, compile_only=True, device=OPENVINO_DEVICE
            )
            self.assertIsInstance(compile_only_model.encoder.model, ov.CompiledModel)
            self.assertIsInstance(compile_only_model.decoder.model, ov.CompiledModel)
            self.assertIsInstance(compile_only_model.postnet.model, ov.CompiledModel)
            self.assertIsInstance(compile_only_model.vocoder.model, ov.CompiledModel)

            outputs = compile_only_model.generate(input_ids=inputs["input_ids"], speaker_embeddings=speaker_embeddings)
            self.assertTrue(torch.equal(loaded_model_outputs, outputs))
            del compile_only_model

        outputs = model.generate(input_ids=inputs["input_ids"], speaker_embeddings=speaker_embeddings)
        self.assertTrue(torch.equal(loaded_model_outputs, outputs))
        del loaded_model
        del model
        gc.collect()

    @pytest.mark.run_slow
    @slow
    def test_load_model_from_hub_private_with_token(self):
        model_id = "optimum-internal-testing/tiny-random-phi-private"
        token = os.environ.get("HF_TOKEN", None)
        if not token:
            self.skipTest("Test requires a token `HF_TOKEN` in the environment variable")

        model = OVModelForCausalLM.from_pretrained(model_id, token=token, revision="openvino", device=OPENVINO_DEVICE)
        self.assertIsInstance(model.config, PretrainedConfig)
        self.assertTrue(model.stateful)

    @parameterized.expand(("", "openvino"))
    def test_loading_with_config_in_root(self, subfolder):
        # config.json file in the root directory and not in the subfolder
        model_id = "sentence-transformers-testing/stsb-bert-tiny-openvino"
        export = subfolder == ""
        # hub model
        OVModelForFeatureExtraction.from_pretrained(
            model_id, subfolder=subfolder, export=export, device=OPENVINO_DEVICE
        )
        with TemporaryDirectory() as tmpdirname:
            local_dir = Path(tmpdirname) / "model"
            snapshot_download(repo_id=model_id, local_dir=local_dir, user_agent=http_user_agent())
            OVModelForFeatureExtraction.from_pretrained(
                local_dir, subfolder=subfolder, export=export, device=OPENVINO_DEVICE
            )

    def test_infer_export_when_loading(self):
        model_id = MODEL_NAMES["phi"]
        model = AutoModelForCausalLM.from_pretrained(model_id)
        with TemporaryDirectory() as tmpdirname:
            model.save_pretrained(Path(tmpdirname) / "original")
            # Load original model and convert
            model = OVModelForCausalLM.from_pretrained(Path(tmpdirname, device=OPENVINO_DEVICE) / "original")
            model.save_pretrained(Path(tmpdirname) / "openvino")
            # Load openvino model
            model = OVModelForCausalLM.from_pretrained(Path(tmpdirname, device=OPENVINO_DEVICE) / "openvino")
        del model
        gc.collect()

    def test_find_files_matching_pattern(self):
        model_id = "echarlaix/tiny-random-PhiForCausalLM"
        pattern = r"(.*)?openvino(.*)?\_model(.*)?.xml$"
        # hub model
        for revision in ("main", "ov", "itrex"):
            ov_files = _find_files_matching_pattern(
                model_id, pattern=pattern, revision=revision, subfolder="openvino" if revision == "itrex" else ""
            )
            self.assertTrue(len(ov_files) == 0 if revision == "main" else len(ov_files) > 0)

        with TemporaryDirectory() as tmpdirname:
            for revision in ("main", "ov", "itrex"):
                local_dir = Path(tmpdirname) / revision
                snapshot_download(
                    repo_id=model_id, local_dir=local_dir, revision=revision, user_agent=http_user_agent()
                )
                ov_files = _find_files_matching_pattern(
                    local_dir, pattern=pattern, revision=revision, subfolder="openvino" if revision == "itrex" else ""
                )
                self.assertTrue(len(ov_files) == 0 if revision == "main" else len(ov_files) > 0)

    @parameterized.expand(("stable-diffusion", "stable-diffusion-openvino"))
    def test_find_files_matching_pattern_sd(self, model_arch):
        pattern = r"(.*)?openvino(.*)?\_model(.*)?.xml$"
        model_id = MODEL_NAMES[model_arch]
        # hub model
        ov_files = _find_files_matching_pattern(model_id, pattern=pattern)
        self.assertTrue(len(ov_files) > 0 if "openvino" in model_id else len(ov_files) == 0)

        with TemporaryDirectory() as tmpdirname:
            local_dir = Path(tmpdirname) / "model"
            snapshot_download(repo_id=model_id, local_dir=local_dir, user_agent=http_user_agent())
            ov_files = _find_files_matching_pattern(local_dir, pattern=pattern)
            self.assertTrue(len(ov_files) > 0 if "openvino" in model_id else len(ov_files) == 0)

    @parameterized.expand(("", "openvino"))
    def test_find_files_matching_pattern_with_config_in_root(self, subfolder):
        # Notably, the model has a config.json file in the root directory and not in the subfolder
        model_id = "sentence-transformers-testing/stsb-bert-tiny-openvino"
        pattern = r"(.*)?openvino(.*)?\_model(.*)?.xml$"
        # hub model
        ov_files = _find_files_matching_pattern(model_id, pattern=pattern, subfolder=subfolder)
        self.assertTrue(len(ov_files) == 1 if subfolder == "openvino" else len(ov_files) == 0)

        with tempfile.TemporaryDirectory() as tmpdirname:
            local_dir = Path(tmpdirname) / "model"
            snapshot_download(repo_id=model_id, local_dir=local_dir, user_agent=http_user_agent())
            ov_files = _find_files_matching_pattern(local_dir, pattern=pattern, subfolder=subfolder)
            self.assertTrue(len(ov_files) == 1 if subfolder == "openvino" else len(ov_files) == 0)

    def test_find_files_matching_pattern_with_quantized_ov_model(self):
        # This model only has "openvino/openvino_model_qint8_quantized.xml" and "openvino/openvino_model_qint8_quantized.bin"
        # We want to ensure that this model is found, so the `export` isn't forced to True
        model_id = "sentence-transformers-testing/stsb-bert-tiny-openvino-quantized-only"
        subfolder = "openvino"
        pattern = r"(.*)?openvino(.*)?\_model(.*)?.xml$"
        # hub model
        ov_files = _find_files_matching_pattern(model_id, pattern=pattern, subfolder=subfolder)
        self.assertTrue(len(ov_files) == 1)

        with tempfile.TemporaryDirectory() as tmpdirname:
            local_dir = Path(tmpdirname) / "model"
            snapshot_download(repo_id=model_id, local_dir=local_dir, user_agent=http_user_agent())
            ov_files = _find_files_matching_pattern(local_dir, pattern=pattern, subfolder=subfolder)
            self.assertTrue(len(ov_files) == 1)

    def test_load_from_hub_onnx_model_and_save(self):
        model_id = "katuni4ka/tiny-random-LlamaForCausalLM-onnx"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer("This is a sample input", return_tensors="pt")
        loaded_model = OVModelForCausalLM.from_pretrained(model_id, from_onnx=True, device=OPENVINO_DEVICE)
        self.assertIsInstance(loaded_model.config, PretrainedConfig)
        # Test that PERFORMANCE_HINT is set to LATENCY by default
        self.assertEqual(loaded_model.ov_config.get("PERFORMANCE_HINT"), "LATENCY")
        self.assertEqual(loaded_model.request.get_compiled_model().get_property("PERFORMANCE_HINT"), "LATENCY")
        loaded_model_outputs = loaded_model(**tokens)

        with TemporaryDirectory() as tmpdirname:
            loaded_model.save_pretrained(tmpdirname)
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(OV_XML_FILE_NAME in folder_contents)
            self.assertTrue(OV_XML_FILE_NAME.replace(".xml", ".bin") in folder_contents)
            model = OVModelForCausalLM.from_pretrained(tmpdirname, device=OPENVINO_DEVICE)
            self.assertEqual(model.use_cache, loaded_model.use_cache)

            compile_only_model = OVModelForCausalLM.from_pretrained(
                tmpdirname, compile_only=True, device=OPENVINO_DEVICE
            )
            self.assertIsInstance(compile_only_model.model, ov.CompiledModel)
            self.assertIsInstance(compile_only_model.request, ov.InferRequest)
            outputs = compile_only_model(**tokens)
            self.assertTrue(torch.equal(loaded_model_outputs.logits, outputs.logits))
            del compile_only_model

        outputs = model(**tokens)
        self.assertTrue(torch.equal(loaded_model_outputs.logits, outputs.logits))
        del loaded_model
        del model
        gc.collect()


class PipelineTest(unittest.TestCase):
    def test_load_model_from_hub(self):
        model_id = "echarlaix/tiny-random-PhiForCausalLM"

        # verify could load both pytorch and openvino model (export argument should automatically infered)
        ov_exported_pipe = optimum_pipeline("text-generation", model_id, revision="pt", accelerator="openvino")
        ov_pipe = optimum_pipeline("text-generation", model_id, revision="ov", accelerator="openvino")
        self.assertIsInstance(ov_exported_pipe.model, OVBaseModel)
        self.assertIsInstance(ov_pipe.model, OVBaseModel)

        with TemporaryDirectory() as tmpdirname:
            ov_exported_pipe.save_pretrained(tmpdirname)
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(OV_XML_FILE_NAME in folder_contents)
            self.assertTrue(OV_XML_FILE_NAME.replace(".xml", ".bin") in folder_contents)
            ov_exported_pipe = optimum_pipeline("text-generation", tmpdirname, accelerator="openvino")
            self.assertIsInstance(ov_exported_pipe.model, OVBaseModel)

        del ov_exported_pipe
        del ov_pipe
        gc.collect()

    def test_seq2seq_load_from_hub(self):
        model_id = "echarlaix/tiny-random-t5"
        # verify could load both pytorch and openvino model (export argument should automatically infered)
        ov_exported_pipe = optimum_pipeline("text2text-generation", model_id, accelerator="openvino")
        ov_pipe = optimum_pipeline("text2text-generation", model_id, revision="ov", accelerator="openvino")
        self.assertIsInstance(ov_exported_pipe.model, OVBaseModel)
        self.assertIsInstance(ov_pipe.model, OVBaseModel)

        with TemporaryDirectory() as tmpdirname:
            ov_exported_pipe.save_pretrained(tmpdirname)
            folder_contents = os.listdir(tmpdirname)
            if not ov_exported_pipe.model.decoder.stateful:
                self.assertTrue(OV_DECODER_WITH_PAST_NAME in folder_contents)
                self.assertTrue(OV_DECODER_WITH_PAST_NAME.replace(".xml", ".bin") in folder_contents)
            ov_exported_pipe = optimum_pipeline("text2text-generation", tmpdirname, accelerator="openvino")
            self.assertIsInstance(ov_exported_pipe.model, OVBaseModel)

        del ov_exported_pipe
        del ov_pipe
        gc.collect()


class OVModelForSequenceClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "albert",
        "bert",
        "convbert",
        "distilbert",
        "electra",
        "flaubert",
        "ibert",
        "roberta",
        "roformer",
        "squeezebert",
        "xlm",
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForSequenceClassification.from_pretrained(
            model_id, export=True, ov_config=F32_CONFIG, device=OPENVINO_DEVICE
        )
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        transformers_model = AutoModelForSequenceClassification.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = "This is a sample input"
        tokens = tokenizer(inputs, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)
        for input_type in ["pt", "np"]:
            tokens = tokenizer(inputs, return_tensors=input_type)
            ov_outputs = ov_model(**tokens)
            self.assertIn("logits", ov_outputs)
            self.assertIsInstance(ov_outputs.logits, TENSOR_ALIAS_TO_TYPE[input_type])
            # Compare tensor outputs
            self.assertTrue(
                torch.allclose(
                    torch.Tensor(ov_outputs.logits),
                    transformers_outputs.logits,
                    atol=1e-4 if model_arch not in ["flaubert", "squeezebert"] else 0.08,
                )
            )
        del transformers_model
        del ov_model
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.run_slow
    @slow
    def test_pipeline(self, model_arch):
        set_seed(SEED)
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForSequenceClassification.from_pretrained(model_id, compile=False, device=OPENVINO_DEVICE)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
        inputs = "This restaurant is awesome"
        outputs = pipe(inputs)
        self.assertTrue(model.is_dynamic)
        self.assertEqual(pipe.device, model.device)
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["label"], str)

        ov_pipe = optimum_pipeline("text-classification", model_id, accelerator="openvino")
        ov_outputs = ov_pipe(inputs)
        atol = 1e-4 if model_arch not in ["flaubert", "squeezebert"] else 0.08
        self.assertTrue(abs(ov_outputs[-1]["score"] - outputs[-1]["score"]) < atol)
        del ov_pipe

        if model_arch == "bert":
            # Test FP16 conversion
            model.half()
            model.to("cpu")
            model.compile()
            outputs = pipe(inputs)
            self.assertGreaterEqual(outputs[0]["score"], 0.0)
            self.assertIsInstance(outputs[0]["label"], str)
            # Test static shapes
            model.reshape(1, 25)
            model.compile()
            outputs = pipe(inputs)
            self.assertTrue(not model.is_dynamic)
            self.assertGreaterEqual(outputs[0]["score"], 0.0)
            self.assertIsInstance(outputs[0]["label"], str)
            # Test that model caching was not automatically enabled for exported model
            openvino_cache_dir = model.model_save_dir / "model_cache"
            self.assertFalse(openvino_cache_dir.is_dir())

        del model
        del pipe
        gc.collect()


class OVModelForQuestionAnsweringIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "bert",
        "distilbert",
        "roberta",
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForQuestionAnswering.from_pretrained(
            model_id, export=True, ov_config=F32_CONFIG, device=OPENVINO_DEVICE
        )
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        transformers_model = AutoModelForQuestionAnswering.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = "This is a sample input"
        tokens = tokenizer(inputs, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)
        for input_type in ["pt", "np"]:
            tokens = tokenizer(inputs, return_tensors=input_type)
            ov_outputs = ov_model(**tokens)
            self.assertIn("start_logits", ov_outputs)
            self.assertIn("end_logits", ov_outputs)
            self.assertIsInstance(ov_outputs.start_logits, TENSOR_ALIAS_TO_TYPE[input_type])
            self.assertIsInstance(ov_outputs.end_logits, TENSOR_ALIAS_TO_TYPE[input_type])
            # Compare tensor outputs
            self.assertTrue(
                torch.allclose(torch.Tensor(ov_outputs.start_logits), transformers_outputs.start_logits, atol=1e-4)
            )
            self.assertTrue(
                torch.allclose(torch.Tensor(ov_outputs.end_logits), transformers_outputs.end_logits, atol=1e-4)
            )
        del ov_model
        del transformers_model
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.run_slow
    @slow
    def test_pipeline(self, model_arch):
        set_seed(SEED)
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForQuestionAnswering.from_pretrained(model_id, device=OPENVINO_DEVICE)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("question-answering", model=model, tokenizer=tokenizer)
        question = "What's my name?"
        context = "My Name is Arthur and I live in Lyon."
        outputs = pipe(question, context)
        self.assertEqual(pipe.device, model.device)
        self.assertGreaterEqual(outputs["score"], 0.0)
        self.assertIsInstance(outputs["answer"], str)
        ov_pipe = optimum_pipeline("question-answering", model_id, accelerator="openvino")
        ov_outputs = ov_pipe(question, context)
        self.assertEqual(outputs["score"], ov_outputs["score"])
        del model
        del ov_pipe
        gc.collect()

    @pytest.mark.run_slow
    @slow
    def test_metric(self):
        model_id = "distilbert-base-cased-distilled-squad"
        set_seed(SEED)
        ov_model = OVModelForQuestionAnswering.from_pretrained(model_id, export=True, device=OPENVINO_DEVICE)
        transformers_model = AutoModelForQuestionAnswering.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        data = load_dataset("squad", split="validation").select(range(50))
        task_evaluator = evaluator("question-answering")
        transformers_pipe = pipeline("question-answering", model=transformers_model, tokenizer=tokenizer)
        ov_pipe = pipeline("question-answering", model=ov_model, tokenizer=tokenizer)
        transformers_metric = task_evaluator.compute(model_or_pipeline=transformers_pipe, data=data, metric="squad")
        ov_metric = task_evaluator.compute(model_or_pipeline=ov_pipe, data=data, metric="squad")
        self.assertEqual(ov_metric["exact_match"], transformers_metric["exact_match"])
        self.assertEqual(ov_metric["f1"], transformers_metric["f1"])
        del transformers_pipe
        del transformers_model
        del ov_pipe
        del ov_model
        gc.collect()


class OVModelForTokenClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "bert",
        "distilbert",
        "roberta",
    )

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForTokenClassification.from_pretrained(
            model_id, export=True, ov_config=F32_CONFIG, device=OPENVINO_DEVICE
        )
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        transformers_model = AutoModelForTokenClassification.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = "This is a sample input"
        tokens = tokenizer(inputs, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)
        for input_type in ["pt", "np"]:
            tokens = tokenizer(inputs, return_tensors=input_type)
            ov_outputs = ov_model(**tokens)
            self.assertIn("logits", ov_outputs)
            self.assertIsInstance(ov_outputs.logits, TENSOR_ALIAS_TO_TYPE[input_type])
            # Compare tensor outputs
            self.assertTrue(torch.allclose(torch.Tensor(ov_outputs.logits), transformers_outputs.logits, atol=1e-4))
        del transformers_model
        del ov_model
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.run_slow
    @slow
    def test_pipeline(self, model_arch):
        set_seed(SEED)
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForTokenClassification.from_pretrained(model_id, device=OPENVINO_DEVICE)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("token-classification", model=model, tokenizer=tokenizer)
        inputs = "My Name is Arthur and I live in"
        outputs = pipe(inputs)
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all(item["score"] > 0.0 for item in outputs))
        ov_pipe = optimum_pipeline("token-classification", model_id, accelerator="openvino")
        ov_outputs = ov_pipe(inputs)
        self.assertEqual(outputs[-1]["score"], ov_outputs[-1]["score"])
        del ov_pipe
        del model
        del pipe
        gc.collect()

    def test_default_token_type_ids(self):
        model_id = MODEL_NAMES["bert"]
        model = OVModelForTokenClassification.from_pretrained(model_id, export=True, device=OPENVINO_DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokens = tokenizer("this is a simple input", return_tensors="np")
        self.assertTrue("token_type_ids" in model.input_names)
        token_type_ids = tokens.pop("token_type_ids")
        outs = model(token_type_ids=token_type_ids, **tokens)
        outs_without_token_type_ids = model(**tokens)
        self.assertTrue(np.allclose(outs.logits, outs_without_token_type_ids.logits))

        tokens["attention_mask"] = None
        with self.assertRaises(Exception) as context:
            _ = model(**tokens)

        self.assertIn("Got unexpected inputs: ", str(context.exception))
        del model
        gc.collect()


class OVModelForFeatureExtractionIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "bert",
        "distilbert",
        "roberta",
        "sentence-transformers-bert",
    )

    if is_transformers_version(">=", "4.51.0"):
        SUPPORTED_ARCHITECTURES += ("qwen3",)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForFeatureExtraction.from_pretrained(
            model_id, export=True, ov_config=F32_CONFIG, device=OPENVINO_DEVICE
        )
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        transformers_model = AutoModel.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = "This is a sample input"
        tokens = tokenizer(inputs, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)
        for input_type in ["pt", "np"]:
            tokens = tokenizer(inputs, return_tensors=input_type)
            ov_outputs = ov_model(**tokens)
            self.assertIn("last_hidden_state", ov_outputs)
            self.assertIsInstance(ov_outputs.last_hidden_state, TENSOR_ALIAS_TO_TYPE[input_type])
            # Compare tensor outputs
            self.assertTrue(
                torch.allclose(
                    torch.Tensor(ov_outputs.last_hidden_state), transformers_outputs.last_hidden_state, atol=1e-4
                )
            )
        del transformers_model
        del ov_model
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.run_slow
    @slow
    def test_pipeline(self, model_arch):
        set_seed(SEED)
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForFeatureExtraction.from_pretrained(model_id, device=OPENVINO_DEVICE)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
        inputs = "My Name is Arthur and I live in"
        outputs = pipe(inputs)
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all(all(isinstance(item, float) for item in row) for row in outputs[0]))
        ov_pipe = optimum_pipeline("feature-extraction", model_id, accelerator="openvino")
        ov_outputs = ov_pipe(inputs)
        self.assertEqual(outputs[-1][-1][-1], ov_outputs[-1][-1][-1])
        del ov_pipe
        del pipe
        del model
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_sentence_transformers_pipeline(self, model_arch):
        """
        Check if we call OVModelForFeatureExtraction passing saved ir-model with outputs
        from Sentence Transformers then an appropriate exception raises.
        """
        model_id = MODEL_NAMES[model_arch]
        with TemporaryDirectory() as tmp_dir:
            save_dir = str(tmp_dir)
            OVSentenceTransformer.from_pretrained(model_id, export=True, device=OPENVINO_DEVICE).save_pretrained(
                save_dir
            )
            with self.assertRaises(Exception) as context:
                OVModelForFeatureExtraction.from_pretrained(save_dir, device=OPENVINO_DEVICE)
            self.assertIn("Please use `OVSentenceTransformer`", str(context.exception))


class OVModelForMaskedLMIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "albert",
        "bert",
        "camembert",
        "convbert",
        "data2vec-text",
        "deberta",
        "deberta-v2",
        "distilbert",
        "electra",
        "esm",
        "flaubert",
        "ibert",
        "mobilebert",
        "mpnet",
        "perceiver_text",
        "rembert",
        "roberta",
        "roformer",
        "squeezebert",
        "xlm",
        "xlm-roberta",
    )

    # accuracy issue, need additional investigation
    if is_transformers_version("<", "4.51.0"):
        SUPPORTED_ARCHITECTURES += ("nystromformer",)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForMaskedLM.from_pretrained(
            model_id, export=True, ov_config=F32_CONFIG, device=OPENVINO_DEVICE
        )
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        set_seed(SEED)
        transformers_model = AutoModelForMaskedLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        inputs = f"This is a sample {tokenizer.mask_token}"
        tokens = tokenizer(inputs, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)
        for input_type in ["pt", "np"]:
            tokens = tokenizer(inputs, return_tensors=input_type)
            ov_outputs = ov_model(**tokens)
            self.assertIn("logits", ov_outputs)
            self.assertIsInstance(ov_outputs.logits, TENSOR_ALIAS_TO_TYPE[input_type])
            # Compare tensor outputs
            self.assertTrue(torch.allclose(torch.Tensor(ov_outputs.logits), transformers_outputs.logits, atol=1e-4))
        del transformers_model
        del ov_model
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        model = OVModelForMaskedLM.from_pretrained(model_id, device=OPENVINO_DEVICE)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer)
        inputs = f"This is a {tokenizer.mask_token}."
        outputs = pipe(inputs)
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all(item["score"] > 0.0 for item in outputs))
        set_seed(SEED)
        ov_pipe = optimum_pipeline("fill-mask", model_id, accelerator="openvino")
        ov_outputs = ov_pipe(inputs)
        self.assertEqual(outputs[-1]["score"], ov_outputs[-1]["score"])
        del ov_pipe
        del pipe
        del model
        gc.collect()


class OVModelForImageClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "beit",
        "convnext",
        # "convnextv2",
        "data2vec-vision",
        "deit",
        "levit",
        "mobilenet_v1",
        "mobilenet_v2",
        "mobilevit",
        "poolformer",
        "perceiver_vision",
        "resnet",
        "segformer",
        "swin",
        "vit",
    )
    TIMM_MODELS = ("timm/pit_s_distilled_224.in1k", "timm/vit_tiny_patch16_224.augreg_in21k")

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForImageClassification.from_pretrained(
            model_id, export=True, ov_config=F32_CONFIG, device=OPENVINO_DEVICE
        )
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        set_seed(SEED)
        transformers_model = AutoModelForImageClassification.from_pretrained(model_id)
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        url = TEST_IMAGE_URL
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=image, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)
        for input_type in ["pt", "np"]:
            inputs = preprocessor(images=image, return_tensors=input_type)
            ov_outputs = ov_model(**inputs)
            self.assertIn("logits", ov_outputs)
            self.assertIsInstance(ov_outputs.logits, TENSOR_ALIAS_TO_TYPE[input_type])
            # Compare tensor outputs
            self.assertTrue(torch.allclose(torch.Tensor(ov_outputs.logits), transformers_outputs.logits, atol=1e-4))
        del transformers_model
        del ov_model
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.run_slow
    @slow
    def test_pipeline(self, model_arch):
        set_seed(SEED)
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForImageClassification.from_pretrained(model_id, device=OPENVINO_DEVICE)
        model.eval()
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        pipe = pipeline("image-classification", model=model, feature_extractor=preprocessor)
        inputs = TEST_IMAGE_URL
        outputs = pipe(inputs)
        self.assertEqual(pipe.device, model.device)
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))
        set_seed(SEED)
        ov_pipe = optimum_pipeline("image-classification", model_id, accelerator="openvino")
        ov_outputs = ov_pipe(inputs)
        self.assertEqual(outputs[-1]["score"], ov_outputs[-1]["score"])
        del ov_pipe
        del model
        del pipe
        gc.collect()

    @parameterized.expand(TIMM_MODELS)
    def test_compare_to_timm(self, model_id):
        ov_model = OVModelForImageClassification.from_pretrained(
            model_id, export=True, ov_config=F32_CONFIG, device=OPENVINO_DEVICE
        )
        self.assertEqual(ov_model.request.get_property("INFERENCE_PRECISION_HINT").to_string(), "f32")
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        timm_model = timm.create_model(model_id, pretrained=True)
        preprocessor = TimmImageProcessor.from_pretrained(model_id)
        url = TEST_IMAGE_URL
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=image, return_tensors="pt")
        with torch.no_grad():
            timm_model.eval()
            timm_outputs = timm_model(inputs["pixel_values"].float())
        for input_type in ["pt", "np"]:
            inputs = preprocessor(images=image, return_tensors=input_type)
            ov_outputs = ov_model(**inputs)
            self.assertIn("logits", ov_outputs)
            self.assertIsInstance(ov_outputs.logits, TENSOR_ALIAS_TO_TYPE[input_type])
            # Compare tensor outputs
            self.assertTrue(torch.allclose(torch.Tensor(ov_outputs.logits), timm_outputs, atol=1e-3))
        gc.collect()

    @parameterized.expand(TIMM_MODELS)
    def test_timm_save_and_infer(self, model_id):
        ov_model = OVModelForImageClassification.from_pretrained(model_id, export=True, device=OPENVINO_DEVICE)
        with TemporaryDirectory() as tmpdirname:
            model_save_path = os.path.join(tmpdirname, "timm_ov_model")
            ov_model.save_pretrained(model_save_path)
            model = OVModelForImageClassification.from_pretrained(model_save_path, device=OPENVINO_DEVICE)
            model(pixel_values=torch.zeros((5, 3, model.config.image_size, model.config.image_size)))
        gc.collect()


class OVModelForAudioClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = (
        "audio-spectrogram-transformer",
        "data2vec-audio",
        "hubert",
        "sew",
        "sew-d",
        "unispeech",
        "unispeech-sat",
        "wavlm",
        "wav2vec2",
        "wav2vec2-conformer",
    )

    def _generate_random_audio_data(self):
        np.random.seed(10)
        t = np.linspace(0, 5.0, int(5.0 * 22050), endpoint=False)
        # generate pure sine wave at 220 Hz
        audio_data = 0.5 * np.sin(2 * np.pi * 220 * t)
        return audio_data

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForAudioClassification.from_pretrained(
            model_id, export=True, ov_config=F32_CONFIG, device=OPENVINO_DEVICE
        )
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        set_seed(SEED)
        transformers_model = AutoModelForAudioClassification.from_pretrained(model_id)
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        inputs = preprocessor(self._generate_random_audio_data(), return_tensors="pt")

        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)

        for input_type in ["pt", "np"]:
            inputs = preprocessor(self._generate_random_audio_data(), return_tensors=input_type)
            ov_outputs = ov_model(**inputs)
            self.assertIn("logits", ov_outputs)
            self.assertIsInstance(ov_outputs.logits, TENSOR_ALIAS_TO_TYPE[input_type])
            # Compare tensor outputs
            self.assertTrue(torch.allclose(torch.Tensor(ov_outputs.logits), transformers_outputs.logits, atol=1e-3))

        del transformers_model
        del ov_model
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @pytest.mark.run_slow
    @slow
    def test_pipeline(self, model_arch):
        set_seed(SEED)
        model_id = MODEL_NAMES[model_arch]
        model = OVModelForAudioClassification.from_pretrained(model_id, device=OPENVINO_DEVICE)
        model.eval()
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        pipe = pipeline("audio-classification", model=model, feature_extractor=preprocessor)
        inputs = [np.random.random(16000)]
        outputs = pipe(inputs)
        self.assertEqual(pipe.device, model.device)
        self.assertTrue(all(item["score"] > 0.0 for item in outputs[0]))
        set_seed(SEED)
        ov_pipe = optimum_pipeline("audio-classification", model_id, accelerator="openvino")
        ov_outputs = ov_pipe(inputs)
        self.assertEqual(outputs[-1][-1]["score"], ov_outputs[-1][-1]["score"])
        del ov_pipe
        del pipe
        del model
        gc.collect()


class OVModelForCTCIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = [
        "data2vec-audio",
        "hubert",
        "sew",
        "sew-d",
        "unispeech",
        "unispeech-sat",
        "wavlm",
        "wav2vec2-hf",
        "wav2vec2-conformer",
    ]

    def _generate_random_audio_data(self):
        np.random.seed(10)
        t = np.linspace(0, 5.0, int(5.0 * 22050), endpoint=False)
        # generate pure sine wave at 220 Hz
        audio_data = 0.5 * np.sin(2 * np.pi * 220 * t)
        return audio_data

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = OVModelForCTC.from_pretrained(MODEL_NAMES["t5"], export=True, device=OPENVINO_DEVICE)

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForCTC.from_pretrained(model_id, export=True, ov_config=F32_CONFIG, device=OPENVINO_DEVICE)
        self.assertIsInstance(ov_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForCTC.from_pretrained(model_id)
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        input_values = processor(self._generate_random_audio_data(), return_tensors="pt")

        with torch.no_grad():
            transformers_outputs = transformers_model(**input_values)

        for input_type in ["pt", "np"]:
            input_values = processor(self._generate_random_audio_data(), return_tensors=input_type)
            ov_outputs = ov_model(**input_values)

            self.assertTrue("logits" in ov_outputs)
            self.assertIsInstance(ov_outputs.logits, TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            self.assertTrue(torch.allclose(torch.Tensor(ov_outputs.logits), transformers_outputs.logits, atol=1e-4))

        del transformers_model
        del ov_model
        gc.collect()


class OVModelForAudioXVectorIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = [
        "data2vec-audio",
        "unispeech-sat",
        "wavlm",
        "wav2vec2-hf",
        "wav2vec2-conformer",
    ]

    def _generate_random_audio_data(self):
        np.random.seed(10)
        t = np.linspace(0, 5.0, int(5.0 * 22050), endpoint=False)
        # generate pure sine wave at 220 Hz
        audio_data = 0.5 * np.sin(2 * np.pi * 220 * t)
        return audio_data

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = OVModelForAudioXVector.from_pretrained(MODEL_NAMES["t5"], export=True, device=OPENVINO_DEVICE)

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForAudioXVector.from_pretrained(
            model_id, export=True, ov_config=F32_CONFIG, device=OPENVINO_DEVICE
        )
        self.assertIsInstance(ov_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForAudioXVector.from_pretrained(model_id)
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        input_values = processor(self._generate_random_audio_data(), return_tensors="pt")

        with torch.no_grad():
            transformers_outputs = transformers_model(**input_values)
        for input_type in ["pt", "np"]:
            input_values = processor(self._generate_random_audio_data(), return_tensors=input_type)
            ov_outputs = ov_model(**input_values)

            self.assertTrue("logits" in ov_outputs)
            self.assertIsInstance(ov_outputs.logits, TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            self.assertTrue(torch.allclose(torch.Tensor(ov_outputs.logits), transformers_outputs.logits, atol=1e-4))
            self.assertTrue(
                torch.allclose(torch.Tensor(ov_outputs.embeddings), transformers_outputs.embeddings, atol=1e-4)
            )

        del transformers_model
        del ov_model
        gc.collect()


class OVModelForAudioFrameClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = [
        "data2vec-audio",
        "unispeech-sat",
        "wavlm",
        "wav2vec2-hf",
        "wav2vec2-conformer",
    ]

    def _generate_random_audio_data(self):
        np.random.seed(10)
        t = np.linspace(0, 5.0, int(5.0 * 22050), endpoint=False)
        # generate pure sine wave at 220 Hz
        audio_data = 0.5 * np.sin(2 * np.pi * 220 * t)
        return audio_data

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = OVModelForAudioFrameClassification.from_pretrained(
                MODEL_NAMES["t5"], export=True, device=OPENVINO_DEVICE
            )

        self.assertIn("only supports the tasks", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForAudioFrameClassification.from_pretrained(
            model_id, export=True, ov_config=F32_CONFIG, device=OPENVINO_DEVICE
        )
        self.assertIsInstance(ov_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForAudioFrameClassification.from_pretrained(model_id)
        processor = AutoFeatureExtractor.from_pretrained(model_id)
        input_values = processor(self._generate_random_audio_data(), return_tensors="pt")

        with torch.no_grad():
            transformers_outputs = transformers_model(**input_values)
        for input_type in ["pt", "np"]:
            input_values = processor(self._generate_random_audio_data(), return_tensors=input_type)
            ov_outputs = ov_model(**input_values)

            self.assertTrue("logits" in ov_outputs)
            self.assertIsInstance(ov_outputs.logits, TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            self.assertTrue(torch.allclose(torch.Tensor(ov_outputs.logits), transformers_outputs.logits, atol=1e-4))

        del transformers_model
        del ov_model
        gc.collect()


class OVModelForCustomTasksIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES_WITH_ATTENTION = ["vit-with-attentions"]
    SUPPORTED_ARCHITECTURES_WITH_HIDDEN_STATES = ["vit-with-hidden-states"]

    def _get_sample_image(self):
        url = TEST_IMAGE_URL
        image = Image.open(requests.get(url, stream=True).raw)
        return image

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_ATTENTION)
    def test_compare_output_attentions(self, model_arch):
        self.skipTest("Skipping until ticket 175062 is resolved.")
        model_id = MODEL_NAMES[model_arch]

        image = self._get_sample_image()
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        inputs = preprocessor(images=image, return_tensors="pt")

        transformers_model = AutoModelForImageClassification.from_pretrained(model_id, attn_implementation="eager")
        transformers_model.eval()
        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs, output_attentions=True)

        ov_model = OVModelForCustomTasks.from_pretrained(model_id, ov_config=F32_CONFIG, device=OPENVINO_DEVICE)
        self.assertIsInstance(ov_model.config, PretrainedConfig)

        for input_type in ["pt", "np"]:
            inputs = preprocessor(images=image, return_tensors=input_type)
            ov_outputs = ov_model(**inputs)
            self.assertIn("logits", ov_outputs)
            self.assertIsInstance(ov_outputs.logits, TENSOR_ALIAS_TO_TYPE[input_type])
            self.assertTrue(torch.allclose(torch.Tensor(ov_outputs.logits), transformers_outputs.logits, atol=1e-4))
            self.assertTrue(len(ov_outputs.attentions) == len(transformers_outputs.attentions))
            for i in range(len(ov_outputs.attentions)):
                self.assertTrue(
                    torch.allclose(
                        torch.Tensor(ov_outputs.attentions[i]),
                        transformers_outputs.attentions[i],
                        atol=1e-4,  # attentions are accurate
                        rtol=1e-4,  # attentions are accurate
                    ),
                    f"Attention mismatch at layer {i}",
                )

        del transformers_model
        del ov_model
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES_WITH_HIDDEN_STATES)
    def test_compare_output_hidden_states(self, model_arch):
        self.skipTest("Skipping until ticket 175062 is resolved.")
        model_id = MODEL_NAMES[model_arch]

        image = self._get_sample_image()
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        inputs = preprocessor(images=image, return_tensors="pt")

        transformers_model = AutoModelForImageClassification.from_pretrained(model_id)
        transformers_model.eval()
        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs, output_hidden_states=True)

        ov_model = OVModelForCustomTasks.from_pretrained(model_id, ov_config=F32_CONFIG, device=OPENVINO_DEVICE)
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        for input_type in ["pt", "np"]:
            inputs = preprocessor(images=image, return_tensors=input_type)
            ov_outputs = ov_model(**inputs)
            self.assertIn("logits", ov_outputs)
            self.assertIsInstance(ov_outputs.logits, TENSOR_ALIAS_TO_TYPE[input_type])
            self.assertTrue(torch.allclose(torch.Tensor(ov_outputs.logits), transformers_outputs.logits, atol=1e-4))
            self.assertTrue(len(ov_outputs.hidden_states) == len(transformers_outputs.hidden_states))
            for i in range(len(ov_outputs.hidden_states)):
                self.assertTrue(
                    torch.allclose(
                        torch.Tensor(ov_outputs.hidden_states[i]),
                        transformers_outputs.hidden_states[i],
                        atol=1e-3,  # hidden states are less accurate
                        rtol=1e-2,  # hidden states are less accurate
                    ),
                    f"Hidden states mismatch at layer {i}",
                )
        del transformers_model
        del ov_model
        gc.collect()


class OVModelForOpenCLIPZeroShortImageClassificationTest(unittest.TestCase):
    OV_MODEL_ID = MODEL_NAMES["open-clip"]
    OV_MODEL_ID_IR = MODEL_NAMES["open-clip-ov"]

    def _get_sample_image(self):
        url = TEST_IMAGE_URL
        image = Image.open(requests.get(url, stream=True).raw)
        return image

    def test_load_from_hub_and_save_model(self):
        loaded_model = OVModelOpenCLIPForZeroShotImageClassification.from_pretrained(
            self.OV_MODEL_ID_IR, device=OPENVINO_DEVICE
        )

        tokenizer = AutoTokenizer.from_pretrained(self.OV_MODEL_ID_IR)
        all_text = ["a dog", "a cat", "a frog"]
        tokens = tokenizer.batch_encode_plus(
            all_text,
            return_tensors="pt",
            max_length=loaded_model.config.text_config.context_length,
            padding="max_length",
            truncation=True,
        ).input_ids

        processor_inputs = {
            "is_train": False,
            "image_size": (loaded_model.config.vision_config.image_size, loaded_model.config.vision_config.image_size),
        }

        processor = open_clip.image_transform(**processor_inputs)
        processed_image = processor(self._get_sample_image()).unsqueeze(0)

        self.assertIsInstance(loaded_model.config, PretrainedConfig)

        loaded_model_outputs = loaded_model(tokens, processed_image)

        with TemporaryDirectory() as tmpdirname:
            loaded_model.save_pretrained(tmpdirname)
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(loaded_model.text_model._all_ov_model_paths["model"] in folder_contents)
            self.assertTrue(
                loaded_model.text_model._all_ov_model_paths["model"].replace(".xml", ".bin") in folder_contents
            )
            self.assertTrue(loaded_model.visual_model._all_ov_model_paths["model"] in folder_contents)
            self.assertTrue(
                loaded_model.visual_model._all_ov_model_paths["model"].replace(".xml", ".bin") in folder_contents
            )
            model = OVModelOpenCLIPForZeroShotImageClassification.from_pretrained(tmpdirname, device=OPENVINO_DEVICE)

        outputs = model(tokens, processed_image)
        self.assertTrue(torch.equal(loaded_model_outputs.logits_per_image, outputs.logits_per_image))
        self.assertTrue(torch.equal(loaded_model_outputs.logits_per_text, outputs.logits_per_text))

        del loaded_model
        del model
        gc.collect()

    def test_compare_output_open_clip(self):
        clip_model, clip_preprocessor = open_clip.create_model_from_pretrained(f"hf-hub:{self.OV_MODEL_ID}")
        clip_tokenizer = open_clip.get_tokenizer(f"hf-hub:{self.OV_MODEL_ID}")

        image = clip_preprocessor(self._get_sample_image()).unsqueeze(0)
        text = clip_tokenizer(["a dog", "a cat", "a frog"])

        with torch.no_grad():
            clip_image_features = clip_model.encode_image(image)
            clip_text_features = clip_model.encode_text(text)

        ov_model = OVModelOpenCLIPForZeroShotImageClassification.from_pretrained(
            self.OV_MODEL_ID, export=True, device=OPENVINO_DEVICE
        )
        ov_outputs = ov_model(text, image)

        self.assertTrue(
            torch.allclose(
                clip_image_features, torch.from_numpy(ov_outputs.vision_model_output["image_features"]), atol=1e-4
            )
        )
        self.assertTrue(
            torch.allclose(
                clip_text_features, torch.from_numpy(ov_outputs.text_model_output["text_features"]), atol=1e-4
            )
        )

        del ov_model
        gc.collect()

    def test_functions(self):
        model = OVModelOpenCLIPForZeroShotImageClassification.from_pretrained(
            self.OV_MODEL_ID, export=True, device=OPENVINO_DEVICE
        )

        tokenizer = AutoTokenizer.from_pretrained(self.OV_MODEL_ID_IR)
        all_text = ["a dog", "a cat", "a frog"]
        tokens = tokenizer.batch_encode_plus(
            all_text,
            return_tensors="pt",
            max_length=model.config.text_config.context_length,
            padding="max_length",
            truncation=True,
        ).input_ids

        processor_inputs = {
            "is_train": False,
            "image_size": (model.config.vision_config.image_size, model.config.vision_config.image_size),
        }

        processor = open_clip.image_transform(**processor_inputs)
        processed_image = processor(self._get_sample_image()).unsqueeze(0)

        model_outputs = model(tokens, processed_image)

        model.to("AUTO")
        self.assertTrue(model.visual_model._device == "AUTO")
        self.assertTrue(model.text_model._device == "AUTO")
        self.assertTrue(model.visual_model.request is None)
        self.assertTrue(model.text_model.request is None)
        res = model(tokens, processed_image)
        self.assertTrue(torch.equal(model_outputs.logits_per_image, res.logits_per_image))

        model.compile()
        self.assertTrue(model.visual_model.request is not None)
        self.assertTrue(model.text_model.request is not None)
        res = model(tokens, processed_image)
        print(model_outputs.logits_per_image, res.logits_per_image)
        self.assertTrue(torch.equal(model_outputs.logits_per_image, res.logits_per_image))

        model.half()
        model.compile()
        res = model(tokens, processed_image)
        print(model_outputs.logits_per_image, res.logits_per_image)
        self.assertTrue(torch.allclose(model_outputs.logits_per_image, res.logits_per_image, atol=1e-2))

        model.reshape(1, -1)
        reshaped_tokens = tokenizer.batch_encode_plus(
            ["a dog"],
            return_tensors="pt",
            max_length=model.config.text_config.context_length,
            padding="max_length",
            truncation=True,
        ).input_ids
        model.compile()
        res = model(reshaped_tokens, processed_image)

        del model
        gc.collect()


class OVModelForSTFeatureExtractionIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ("st-bert", "st-mpnet")

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVSentenceTransformer.from_pretrained(
            model_id, export=True, ov_config=F32_CONFIG, device=OPENVINO_DEVICE
        )
        self.assertIsInstance(ov_model.config, PretrainedConfig)
        self.assertTrue(hasattr(ov_model, "encode"))
        st_model = SentenceTransformer(model_id)
        sentences = ["This is an example sentence", "Each sentence is converted"]
        st_embeddings = st_model.encode(sentences)
        ov_embeddings = ov_model.encode(sentences)
        # Compare tensor outputs
        self.assertTrue(np.allclose(ov_embeddings, st_embeddings, atol=1e-4))
        del st_embeddings
        del ov_model
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_sentence_transformers_save_and_infer(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        ov_model = OVSentenceTransformer.from_pretrained(
            model_id, export=True, ov_config=F32_CONFIG, device=OPENVINO_DEVICE
        )
        with TemporaryDirectory() as tmpdirname:
            model_save_path = os.path.join(tmpdirname, "sentence_transformers_ov_model")
            ov_model.save_pretrained(model_save_path)
            model = OVSentenceTransformer.from_pretrained(model_save_path, device=OPENVINO_DEVICE)
            sentences = ["This is an example sentence", "Each sentence is converted"]
            model.encode(sentences)
        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @unittest.skipIf(not _langchain_hf_available, reason="langchain not installed")
    def test_langchain(self, model_arch):
        from langchain_huggingface import HuggingFaceEmbeddings

        model_id = MODEL_NAMES[model_arch]
        model_kwargs = {"device": "cpu", "backend": "openvino"}

        embedding = HuggingFaceEmbeddings(
            model_name=model_id,
            model_kwargs=model_kwargs,
        )
        output = embedding.embed_query("foo bar")
        self.assertTrue(len(output) > 0)


class OVLangchainTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ("gpt2",)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @unittest.skipIf(not _langchain_hf_available, reason="langchain not installed")
    def test_huggingface_pipeline_streaming(self, model_arch):
        from langchain_huggingface import HuggingFacePipeline

        model_id = MODEL_NAMES[model_arch]

        hf_pipe = HuggingFacePipeline.from_model_id(
            model_id=model_id,
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 10},
            backend="openvino",
        )
        self.assertIsInstance(hf_pipe.pipeline.model, OVBaseModel)

        generator = hf_pipe.stream("Q: How do you say 'hello' in German? A:'", stop=["."])

        self.assertIsInstance(generator, Generator)

        stream_results_string = ""
        for chunk in generator:
            self.assertIsInstance(chunk, str)
            stream_results_string = chunk

        self.assertTrue(len(stream_results_string.strip()) > 1)

        del hf_pipe
        gc.collect()


class OVSamIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ["sam"]
    TASK = "feature-extraction"
    IMAGE_URL = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        from optimum.intel.openvino.modeling_sam import OVSamPromptEncoder, OVSamVisionEncoder

        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVSamModel.from_pretrained(model_id, export=True, ov_config=F32_CONFIG, device=OPENVINO_DEVICE)
        processor = get_preprocessor(model_id)

        self.assertIsInstance(ov_model.vision_encoder, OVSamVisionEncoder)
        self.assertIsInstance(ov_model.prompt_encoder_mask_decoder, OVSamPromptEncoder)
        self.assertIsInstance(ov_model.config, PretrainedConfig)

        input_points = [[[450, 600]]]
        IMAGE = Image.open(
            requests.get(
                self.IMAGE_URL,
                stream=True,
            ).raw
        ).convert("RGB")
        inputs = processor(IMAGE, input_points=input_points, return_tensors="pt")

        transformers_model = SamModel.from_pretrained(model_id)

        # test end-to-end inference
        ov_outputs = ov_model(**inputs)

        self.assertTrue("pred_masks" in ov_outputs)
        self.assertIsInstance(ov_outputs.pred_masks, torch.Tensor)
        self.assertTrue("iou_scores" in ov_outputs)
        self.assertIsInstance(ov_outputs.iou_scores, torch.Tensor)

        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)
        # Compare tensor outputs
        self.assertTrue(torch.allclose(ov_outputs.pred_masks, transformers_outputs.pred_masks, atol=1e-4))
        self.assertTrue(torch.allclose(ov_outputs.iou_scores, transformers_outputs.iou_scores, atol=1e-4))

        # test separated image features extraction
        pixel_values = inputs.pop("pixel_values")
        features = transformers_model.get_image_embeddings(pixel_values)
        ov_features = ov_model.get_image_features(pixel_values)
        self.assertTrue(torch.allclose(ov_features, features, atol=1e-4))
        ov_outputs = ov_model(**inputs, image_embeddings=ov_features)
        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs, image_embeddings=features)
        # Compare tensor outputs
        self.assertTrue(torch.allclose(ov_outputs.pred_masks, transformers_outputs.pred_masks, atol=1e-4))
        self.assertTrue(torch.allclose(ov_outputs.iou_scores, transformers_outputs.iou_scores, atol=1e-4))

        del transformers_model
        del ov_model

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_reshape(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVSamModel.from_pretrained(model_id, export=True, ov_config=F32_CONFIG, device=OPENVINO_DEVICE)
        processor = get_preprocessor(model_id)
        self.assertTrue(ov_model.is_dynamic)
        input_points = [[[450, 600]]]
        IMAGE = Image.open(
            requests.get(
                self.IMAGE_URL,
                stream=True,
            ).raw
        ).convert("RGB")
        inputs = processor(IMAGE, input_points=input_points, return_tensors="pt")
        ov_dyn_outputs = ov_model(**inputs)
        ov_model.reshape(*inputs["input_points"].shape[:-1])
        self.assertFalse(ov_model.is_dynamic)
        self.assertIsNone(ov_model.vision_encoder.request)
        self.assertIsNone(ov_model.prompt_encoder_mask_decoder.request)
        ov_stat_outputs = ov_model(**inputs)
        # Compare tensor outputs
        self.assertTrue(torch.allclose(ov_dyn_outputs.pred_masks, ov_stat_outputs.pred_masks, atol=1e-4))
        self.assertTrue(torch.allclose(ov_dyn_outputs.iou_scores, ov_stat_outputs.iou_scores, atol=1e-4))

        del ov_model
        gc.collect()


class OVModelForZeroShotImageClassificationIntegrationTest(unittest.TestCase):
    SUPPORTED_ARCHITECTURES = ["clip", "siglip"]
    TASK = "zero-shot-image-classification"
    IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_id = MODEL_NAMES[model_arch]
        set_seed(SEED)
        ov_model = OVModelForZeroShotImageClassification.from_pretrained(
            model_id, export=True, ov_config=F32_CONFIG, device=OPENVINO_DEVICE
        )
        processor = get_preprocessor(model_id)

        self.assertIsInstance(ov_model.config, PretrainedConfig)

        IMAGE = Image.open(
            requests.get(
                self.IMAGE_URL,
                stream=True,
            ).raw
        ).convert("RGB")
        labels = ["a photo of a cat", "a photo of a dog"]
        inputs = processor(images=IMAGE, text=labels, return_tensors="pt")

        transformers_model = AutoModelForZeroShotImageClassification.from_pretrained(model_id)

        # test end-to-end inference
        ov_outputs = ov_model(**inputs)

        self.assertTrue("logits_per_image" in ov_outputs)
        self.assertIsInstance(ov_outputs.logits_per_image, torch.Tensor)
        self.assertTrue("logits_per_text" in ov_outputs)
        self.assertIsInstance(ov_outputs.logits_per_text, torch.Tensor)
        self.assertTrue("text_embeds" in ov_outputs)
        self.assertIsInstance(ov_outputs.text_embeds, torch.Tensor)
        self.assertTrue("image_embeds" in ov_outputs)
        self.assertIsInstance(ov_outputs.image_embeds, torch.Tensor)

        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)
        # Compare tensor outputs
        self.assertTrue(torch.allclose(ov_outputs.logits_per_image, transformers_outputs.logits_per_image, atol=1e-4))
        self.assertTrue(torch.allclose(ov_outputs.logits_per_text, transformers_outputs.logits_per_text, atol=1e-4))
        self.assertTrue(torch.allclose(ov_outputs.text_embeds, transformers_outputs.text_embeds, atol=1e-4))
        self.assertTrue(torch.allclose(ov_outputs.image_embeds, transformers_outputs.image_embeds, atol=1e-4))

        del transformers_model
        del ov_model
        gc.collect()
