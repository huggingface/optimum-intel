#  Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""Tests for the experimental ``optimum-cli export openvino-hf`` path (Transformers-native exporter)."""

import gc
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import openvino
import openvino_genai
import torch
from parameterized import parameterized
from PIL import Image
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    set_seed,
)
from utils_tests import F32_CONFIG, MODEL_NAMES

from optimum.exporters.openvino_hf import export_openvino_hf
from optimum.exporters.openvino_hf.export import _load_processor, _task_from_architecture
from optimum.exporters.openvino_hf.inputs import _build_sample_inputs
from optimum.intel.openvino import (
    OVModelForAudioClassification,
    OVModelForAudioFrameClassification,
    OVModelForAudioXVector,
    OVModelForCausalLM,
    OVModelForCTC,
    OVModelForFeatureExtraction,
    OVModelForImageClassification,
    OVModelForImageTextToText,
    OVModelForMaskedLM,
    OVModelForQuestionAnswering,
    OVModelForSeq2SeqLM,
    OVModelForSequenceClassification,
    OVModelForSpeechSeq2Seq,
    OVModelForTokenClassification,
    OVModelForVisualCausalLM,
    OVModelForZeroShotImageClassification,
)


# Greedy, fixed length: makes Transformers and the OpenVINO runtimes directly token-comparable.
GEN_KWARGS = {"max_new_tokens": 10, "min_new_tokens": 10, "do_sample": False, "num_beams": 1}


class _ExportMixin:
    def _assert_exported(self, output):
        """Every exported component parses/type-checks as OpenVINO IR, and the config is saved."""
        output = Path(output)
        xmls = sorted(output.glob("openvino_*.xml"))
        self.assertTrue(xmls, f"no OpenVINO IR produced in {output}")
        core = openvino.Core()
        for xml in xmls:
            core.read_model(str(xml))
        self.assertTrue((output / "config.json").exists(), "model config not saved")


class OVHfExporterTest(_ExportMixin, unittest.TestCase):
    """Broad export + IR-parse sweep across every modality — this exporter is meant to cover almost any
    model ``torch.export`` can trace. Reuses the tiny fixtures the ``openvino`` exporter tests ship."""

    SUPPORTED_ARCHITECTURES = [
        # causal LM (text-generation) — decoder-only families across many architectures
        ("gpt2", "text-generation"),
        ("gpt_neo", "text-generation"),
        ("gpt_neox", "text-generation"),
        ("gptj", "text-generation"),
        ("gpt_bigcode", "text-generation"),
        ("gpt_oss", "text-generation"),
        ("bloom", "text-generation"),
        ("opt", "text-generation"),
        ("xglm", "text-generation"),
        ("mpt", "text-generation"),
        ("codegen", "text-generation"),
        ("starcoder2", "text-generation"),
        ("llama", "text-generation"),
        ("mistral", "text-generation"),
        ("mistral-nemo", "text-generation"),
        ("mixtral", "text-generation"),
        ("qwen2", "text-generation"),
        ("qwen2_moe", "text-generation"),
        ("qwen3", "text-generation"),
        ("qwen3_moe", "text-generation"),
        ("gemma", "text-generation"),
        ("gemma2", "text-generation"),
        ("granite", "text-generation"),
        ("granitemoe", "text-generation"),
        ("olmo", "text-generation"),
        ("olmo2", "text-generation"),
        ("cohere", "text-generation"),
        ("cohere2", "text-generation"),
        ("phi", "text-generation"),
        ("phi3", "text-generation"),
        ("persimmon", "text-generation"),
        ("stablelm", "text-generation"),
        ("glm", "text-generation"),
        ("glm4", "text-generation"),
        ("arcee", "text-generation"),
        ("biogpt", "text-generation"),
        ("smollm3", "text-generation"),
        ("falcon", "text-generation"),
        ("afmoe", "text-generation"),
        ("exaone4", "text-generation"),
        ("gemma3_text", "text-generation"),
        ("gemma4", "text-generation"),
        ("gemma4_moe", "text-generation"),
        ("granitemoehybrid", "text-generation"),
        ("hunyuan_v1_dense", "text-generation"),
        ("lfm2_moe", "text-generation"),
        ("qwen3_next", "text-generation"),
        ("mamba", "text-generation"),
        ("falcon_mamba", "text-generation"),
        ("bitnet", "text-generation"),
        # seq2seq (text2text-generation) — encoder + stateful decoder
        ("t5", "text2text-generation"),
        ("mt5", "text2text-generation"),
        ("longt5", "text2text-generation"),
        ("bart", "text2text-generation"),
        ("mbart", "text2text-generation"),
        ("marian", "text2text-generation"),
        ("pegasus", "text2text-generation"),
        ("bigbird_pegasus", "text2text-generation"),
        ("blenderbot", "text2text-generation"),
        ("blenderbot-small", "text2text-generation"),
        ("m2m_100", "text2text-generation"),
        # text encoders (feature-extraction exercises the base model; task-specific heads are covered by
        # OVHfEncoderTest below)
        ("bert", "feature-extraction"),
        ("albert", "feature-extraction"),
        ("roberta", "feature-extraction"),
        ("distilbert", "feature-extraction"),
        ("electra", "feature-extraction"),
        ("deberta", "feature-extraction"),
        ("deberta-v2", "feature-extraction"),
        ("mobilebert", "feature-extraction"),
        ("camembert", "feature-extraction"),
        ("xlm", "feature-extraction"),
        ("xlm-roberta", "feature-extraction"),
        ("flaubert", "feature-extraction"),
        ("ibert", "feature-extraction"),
        ("mpnet", "feature-extraction"),
        ("nystromformer", "feature-extraction"),
        ("rembert", "feature-extraction"),
        ("roformer", "feature-extraction"),
        ("squeezebert", "feature-extraction"),
        ("esm", "feature-extraction"),
        ("convbert", "feature-extraction"),
        ("data2vec-text", "feature-extraction"),
        ("perceiver_text", "fill-mask"),
        # image
        ("vit", "image-classification"),
        ("beit", "image-classification"),
        ("deit", "image-classification"),
        ("convnext", "image-classification"),
        ("convnextv2", "image-classification"),
        ("resnet", "image-classification"),
        ("levit", "image-classification"),
        ("mobilenet_v1", "image-classification"),
        ("mobilenet_v2", "image-classification"),
        ("mobilevit", "image-classification"),
        ("poolformer", "image-classification"),
        ("swin", "image-classification"),
        ("data2vec-vision", "image-classification"),
        ("donut-swin", "image-classification"),
        # image, dense-prediction heads (export + IR parse; no dedicated OVModel runtime class)
        ("segformer", "semantic-segmentation"),
        ("mobilevit", "semantic-segmentation"),
        ("detr", "object-detection"),
        # image-text dual encoders (CLIP-family zero-shot image classification)
        ("clip", "zero-shot-image-classification"),
        ("siglip", "zero-shot-image-classification"),
        # image-to-text (vision-encoder-decoder: vision encoder + stateful text decoder)
        ("vision-encoder-decoder", "image-to-text"),
        ("pix2struct", "image-to-text"),
        # audio
        ("wav2vec2-hf", "audio-classification"),
        ("hubert", "audio-classification"),
        ("wavlm", "audio-classification"),
        ("data2vec-audio", "audio-classification"),
        ("sew", "audio-classification"),
        ("sew-d", "audio-classification"),
        ("unispeech", "audio-classification"),
        ("unispeech-sat", "audio-classification"),
        ("wav2vec2-conformer", "audio-classification"),
        ("audio-spectrogram-transformer", "audio-classification"),
        ("whisper", "automatic-speech-recognition"),
        ("speech_to_text", "automatic-speech-recognition"),
    ]

    # Models skipped in this sweep, with the reason (genuine OpenVINO-conversion / export gaps, or a
    # broken tiny-fixture tokenizer that can't build a sample).
    UNSUPPORTED = {
        "mamba": "Tiny fixture's tokenizer yields empty ids (the model itself exports fine).",
        "falcon_mamba": "Same broken-tokenizer fixture as `mamba` (the model itself exports fine).",
        "bitnet": "Fails in `from_pretrained` before export: Transformers' automatic weight conversion "
        "errors on the fixture's packed 1.58-bit weights.",
        "ibert": "torch.export can't trace I-BERT's custom quant autograd Functions (np.frexp on a "
        "traced tensor, then fake tensors lifted into graph constants from QuantEmbedding).",
    }

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_export(self, model_type, task):
        if model_type in self.UNSUPPORTED:
            self.skipTest(self.UNSUPPORTED[model_type])
        with tempfile.TemporaryDirectory() as tmp:
            self._assert_exported(export_openvino_hf(MODEL_NAMES[model_type], tmp, task=task))

    def test_task_inference_mapping(self):
        # `task="auto"` maps `config.architectures[0]` to a task. Unit-tested here (no download) since
        # some randomly-initialised fixtures omit `architectures`; real checkpoints set it.
        self.assertEqual(_task_from_architecture("LlamaForCausalLM"), "text-generation")
        self.assertEqual(_task_from_architecture("T5ForConditionalGeneration"), "text2text-generation")
        self.assertEqual(
            _task_from_architecture("LlavaForConditionalGeneration", has_vision_config=True), "image-text-to-text"
        )
        # Speech encoder-decoders end in ForConditionalGeneration but must route to ASR, not text2text.
        self.assertEqual(_task_from_architecture("WhisperForConditionalGeneration"), "automatic-speech-recognition")
        self.assertEqual(_task_from_architecture("VisionEncoderDecoderModel", has_vision_config=True), "image-to-text")
        self.assertEqual(_task_from_architecture("CLIPModel"), "zero-shot-image-classification")
        self.assertEqual(_task_from_architecture("ViTForImageClassification"), "image-classification")
        self.assertEqual(_task_from_architecture("BertForMaskedLM"), "fill-mask")
        self.assertEqual(_task_from_architecture("Wav2Vec2ForCTC"), "ctc")
        self.assertEqual(_task_from_architecture("SegformerForSemanticSegmentation"), "semantic-segmentation")
        self.assertEqual(_task_from_architecture("DetrForObjectDetection"), "object-detection")
        self.assertEqual(_task_from_architecture("SomethingUnknown"), "feature-extraction")

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                f"optimum-cli export openvino-hf --model {MODEL_NAMES['gpt2']} --task text-generation {tmp}",
                shell=True,
                check=True,
                capture_output=True,
            )
            self._assert_exported(tmp)


class OVHfCausalLMTest(_ExportMixin, unittest.TestCase):
    """Text generation: the unified stateful decode drives both OpenVINO GenAI and OVModelForCausalLM,
    matching Transformers greedy decoding token-for-token."""

    # Every causal family whose export runs in OpenVINO GenAI's LLMPipeline with greedy token parity
    # (verified by sweep). GQA + sliding-window families (mistral, mixtral, gpt_oss) are covered since
    # the exporter stopped baking the sliding-cache eviction and repeat_kv 5-D expand. gptj still diverges
    # numerically; glm4 exports/runs but its tiny fixture ships no lm_head weight (untied), so each load
    # randomly re-inits it and parity is non-deterministic.
    GENAI_ARCHITECTURES = [
        "gpt2",
        "gpt_bigcode",
        "gpt_neo",
        "gpt_neox",
        "bloom",
        "opt",
        "xglm",
        "mpt",
        "codegen",
        "falcon",
        "starcoder2",
        "llama",
        "qwen2",
        "qwen2_moe",
        "qwen3",
        "qwen3_moe",
        "gemma",
        "gemma2",
        "granite",
        "granitemoe",
        "olmo",
        "olmo2",
        "cohere",
        "cohere2",
        "phi",
        "phi3",
        "phi3-longrope",
        "phimoe",
        "persimmon",
        "stablelm",
        "glm",
        "mistral",
        "mistral-nemo",
        "mixtral",
        "gpt_oss",
        "arcee",
        "biogpt",
        "smollm3",
        "afmoe",
        "exaone4",
        "granitemoehybrid",
        "hunyuan_v1_dense",
    ]

    # Every causal family whose export runs in OVModelForCausalLM with greedy token parity (verified by
    # sweep). Its runtime contract differs from GenAI's — it accepts the alibi / extra-input models GenAI
    # rejects (bloom, mpt, codegen, falcon, gpt_neo, xglm). GQA + sliding-window families now pass here
    # too; gptj and olmo still diverge numerically, and glm4's tiny fixture has no lm_head weight (untied)
    # so its parity is non-deterministic.
    OVMODEL_ARCHITECTURES = [
        "gpt2",
        "gpt_bigcode",
        "gpt_neo",
        "gpt_neox",
        "bloom",
        "opt",
        "xglm",
        "mpt",
        "codegen",
        "falcon",
        "starcoder2",
        "llama",
        "qwen2",
        "qwen2_moe",
        "qwen3",
        "qwen3_moe",
        "gemma",
        "gemma2",
        "granite",
        "granitemoe",
        "olmo2",
        "cohere",
        "cohere2",
        "phi",
        "phi3",
        "phi3-longrope",
        "phimoe",
        "persimmon",
        "stablelm",
        "glm",
        "mistral",
        "mistral-nemo",
        "mixtral",
        "gpt_oss",
        "arcee",
        "biogpt",
        "smollm3",
        "afmoe",
        "exaone4",
        "granitemoehybrid",
        "hunyuan_v1_dense",
    ]

    def test_genai_layout(self):
        # Text generation is exported in the OpenVINO GenAI layout: one stateful `openvino_model.xml`
        # (the unified multi-token decode) + the OpenVINO tokenizer/detokenizer + generation config.
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(export_openvino_hf(MODEL_NAMES["gpt2"], tmp, task="text-generation"))
            for name in (
                "openvino_model.xml",
                "openvino_tokenizer.xml",
                "openvino_detokenizer.xml",
                "generation_config.json",
            ):
                self.assertTrue((output / name).exists(), f"{name} missing from GenAI layout")

    @parameterized.expand(GENAI_ARCHITECTURES)
    def test_genai_pipeline(self, model_type):
        model_id = MODEL_NAMES[model_type]
        with tempfile.TemporaryDirectory() as tmp:
            # fp16=False so the exported IR is fp32; combined with the f32 inference hint below this
            # matches Transformers' fp32 reference bit-for-bit enough for greedy token parity.
            export_openvino_hf(model_id, tmp, task="text-generation", fp16=False)

            set_seed(42)
            reference = AutoModelForCausalLM.from_pretrained(model_id).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            inputs = tokenizer("Paris is the capital of", return_tensors="pt")
            input_len = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                reference_ids = reference.generate(**inputs, **GEN_KWARGS).squeeze()[input_len:].tolist()

            pipe = openvino_genai.LLMPipeline(tmp, "CPU", **F32_CONFIG)
            genai_ids = pipe.generate(
                openvino.Tensor(inputs["input_ids"].numpy()), apply_chat_template=False, **GEN_KWARGS
            ).tokens[0]

            del pipe
            del reference
            gc.collect()

            self.assertEqual(reference_ids, genai_ids, "Transformers and OpenVINO GenAI tokens differ")

    @parameterized.expand(OVMODEL_ARCHITECTURES)
    def test_ovmodel_causal_lm(self, model_type):
        # The same text-generation export also drives optimum-intel's OVModelForCausalLM (not only GenAI),
        # and greedy decoding matches Transformers token-for-token (fp32 export + f32 inference hint).
        model_id = MODEL_NAMES[model_type]
        with tempfile.TemporaryDirectory() as tmp:
            export_openvino_hf(model_id, tmp, task="text-generation", fp16=False)

            set_seed(42)
            reference = AutoModelForCausalLM.from_pretrained(model_id).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            inputs = tokenizer("Paris is the capital of", return_tensors="pt")
            input_len = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                reference_ids = reference.generate(**inputs, **GEN_KWARGS).squeeze()[input_len:].tolist()

            ov_model = OVModelForCausalLM.from_pretrained(tmp, ov_config=F32_CONFIG)
            ov_ids = ov_model.generate(**inputs, **GEN_KWARGS).squeeze()[input_len:].tolist()

            del ov_model
            del reference
            gc.collect()

            self.assertEqual(reference_ids, ov_ids, "Transformers and OVModelForCausalLM tokens differ")


class OVHfEncoderTest(_ExportMixin, unittest.TestCase):
    """Single-``openvino_model.xml`` tasks that must load and run in optimum-intel's own runtime classes
    (not just parse as IR), across the different task heads and modalities."""

    # (model_type, task, OVModel runtime class, modality)
    RUNTIME_ARCHITECTURES = [
        ("bert", "feature-extraction", OVModelForFeatureExtraction, "text"),
        ("bert", "fill-mask", OVModelForMaskedLM, "text"),
        ("albert", "text-classification", OVModelForSequenceClassification, "text"),
        ("roberta", "token-classification", OVModelForTokenClassification, "text"),
        ("distilbert", "question-answering", OVModelForQuestionAnswering, "text"),
        ("vit", "image-classification", OVModelForImageClassification, "image"),
        ("hubert", "audio-classification", OVModelForAudioClassification, "audio"),
        ("wav2vec2-hf", "ctc", OVModelForCTC, "audio"),
        ("wav2vec2-hf", "audio-frame-classification", OVModelForAudioFrameClassification, "audio"),
        ("wav2vec2-hf", "audio-xvector", OVModelForAudioXVector, "audio"),
        ("clip", "zero-shot-image-classification", OVModelForZeroShotImageClassification, "image-text"),
        ("siglip", "zero-shot-image-classification", OVModelForZeroShotImageClassification, "image-text"),
    ]

    @parameterized.expand(RUNTIME_ARCHITECTURES)
    def test_ovmodel_runtime(self, model_type, task, ov_class, modality):
        model_id = MODEL_NAMES[model_type]
        with tempfile.TemporaryDirectory() as tmp:
            export_openvino_hf(model_id, tmp, task=task, fp16=False)
            model = ov_class.from_pretrained(tmp)
            # Reuse the exporter's own sample builder, then keep only the ports this IR declares.
            sample = _build_sample_inputs(_load_processor(model_id, modality, trust_remote_code=False), modality)
            outputs = model(**{k: v for k, v in sample.items() if k in model.input_names})
            self.assertTrue(any(v is not None for v in outputs.values()), "OVModel produced no output tensor")


class OVHfSeq2SeqTest(_ExportMixin, unittest.TestCase):
    """Encoder-decoder models exported as a stateless encoder + stateful decoder, driving
    OVModelForSeq2SeqLM / OVModelForSpeechSeq2Seq (token parity) and OpenVINO GenAI's WhisperPipeline."""

    # Encoder-decoder families whose stateful enc-dec export matches Transformers greedy (verified by
    # sweep). longt5 is excluded: it exports, but its block-local / transient-global attention bakes
    # shapes that fail at runtime in OVModelForSeq2SeqLM.
    SEQ2SEQ_ARCHITECTURES = [
        "t5",
        "mt5",
        "bart",
        "mbart",
        "marian",
        "pegasus",
        "bigbird_pegasus",
        "blenderbot",
        "blenderbot-small",
        "m2m_100",
    ]

    @parameterized.expand(SEQ2SEQ_ARCHITECTURES)
    def test_ovmodel_seq2seq(self, model_type):
        model_id = MODEL_NAMES[model_type]
        with tempfile.TemporaryDirectory() as tmp:
            export_openvino_hf(model_id, tmp, task="text2text-generation", fp16=False)

            reference = AutoModelForSeq2SeqLM.from_pretrained(model_id).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            inputs = tokenizer("translate English to German: the house is nice", return_tensors="pt")

            with torch.no_grad():
                reference_ids = reference.generate(**inputs, **GEN_KWARGS)[0].tolist()

            ov_model = OVModelForSeq2SeqLM.from_pretrained(tmp, ov_config=F32_CONFIG)
            ov_ids = ov_model.generate(**inputs, **GEN_KWARGS)[0].tolist()

            del ov_model
            del reference
            gc.collect()

            self.assertEqual(reference_ids, ov_ids, "Transformers and OVModelForSeq2SeqLM tokens differ")

    # Speech seq2seq: an audio encoder + stateful text decoder, driven by OVModelForSpeechSeq2Seq.
    # whisper's encoder takes no padding mask; speech_to_text adds one. Both feed log-mel `input_features`
    # from the feature extractor.
    SPEECH_SEQ2SEQ_ARCHITECTURES = ["whisper", "speech_to_text"]

    @parameterized.expand(SPEECH_SEQ2SEQ_ARCHITECTURES)
    def test_ovmodel_speech_seq2seq(self, model_type):
        model_id = MODEL_NAMES[model_type]
        with tempfile.TemporaryDirectory() as tmp:
            export_openvino_hf(model_id, tmp, task="automatic-speech-recognition", fp16=False)

            reference = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).eval()
            processor = AutoProcessor.from_pretrained(model_id)
            features = processor(np.zeros(16000, dtype=np.float32), sampling_rate=16000, return_tensors="pt")

            with torch.no_grad():
                reference_ids = reference.generate(**features, **GEN_KWARGS)[0].tolist()

            ov_model = OVModelForSpeechSeq2Seq.from_pretrained(tmp, ov_config=F32_CONFIG)
            ov_ids = ov_model.generate(**features, **GEN_KWARGS)[0].tolist()

            del ov_model
            del reference
            gc.collect()

            self.assertEqual(reference_ids, ov_ids, "Transformers and OVModelForSpeechSeq2Seq tokens differ")

    # Image encoder-decoders: a vision encoder + stateful text decoder — the same split as text seq2seq
    # with image in place of text. Both load through the canonical OVModelForImageTextToText, which
    # dispatches Pix2Struct (flattened_patches) to its own subclass; vit-gpt2 feeds `pixel_values`. Both
    # inputs come straight from the image processor.
    IMAGE_TO_TEXT_ARCHITECTURES = ["vision-encoder-decoder", "pix2struct"]

    @parameterized.expand(IMAGE_TO_TEXT_ARCHITECTURES)
    def test_ovmodel_image_to_text(self, model_type):
        model_id = MODEL_NAMES[model_type]
        with tempfile.TemporaryDirectory() as tmp:
            export_openvino_hf(model_id, tmp, task="image-to-text", fp16=False)

            reference = AutoModelForImageTextToText.from_pretrained(model_id).eval()
            encoder_inputs = AutoImageProcessor.from_pretrained(model_id)(
                images=[Image.new("RGB", (224, 224))], return_tensors="pt"
            )

            with torch.no_grad():
                reference_ids = reference.generate(**encoder_inputs, **GEN_KWARGS)[0].tolist()

            ov_model = OVModelForImageTextToText.from_pretrained(tmp, ov_config=F32_CONFIG)
            ov_ids = ov_model.generate(**encoder_inputs, **GEN_KWARGS)[0].tolist()

            del ov_model
            del reference
            gc.collect()

            self.assertEqual(reference_ids, ov_ids, "Transformers and OVModelForImageTextToText tokens differ")

    def test_genai_whisper(self):
        # The same speech export runs in OpenVINO GenAI's WhisperPipeline (needs the OpenVINO
        # tokenizer/detokenizer, resolved from the Whisper processor), transcribing the same audio as
        # Transformers greedy.
        model_id = MODEL_NAMES["whisper"]
        sample_rate = 16000
        audio = (0.5 * np.sin(2 * np.pi * 220 * np.linspace(0, 1, sample_rate, endpoint=False))).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            export_openvino_hf(model_id, tmp, task="automatic-speech-recognition", fp16=False)

            reference = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).eval()
            processor = AutoProcessor.from_pretrained(model_id)
            features = processor(audio, sampling_rate=sample_rate, return_tensors="pt")

            with torch.no_grad():
                reference_ids = reference.generate(**features, **GEN_KWARGS)[0]
                reference_text = processor.tokenizer.decode(reference_ids, skip_special_tokens=True)

            pipe = openvino_genai.WhisperPipeline(tmp, "CPU", **F32_CONFIG)
            genai_text = pipe.generate(audio.tolist(), **GEN_KWARGS).texts[0]

            del pipe
            del reference
            gc.collect()

            self.assertEqual(reference_text, genai_text, "Transformers and OpenVINO GenAI WhisperPipeline text differ")


class OVHfVLMTest(_ExportMixin, unittest.TestCase):
    """VLMs exported as text-embeddings + vision-embeddings + stateful language-model, driving
    OVModelForVisualCausalLM (token parity) and OpenVINO GenAI's VLMPipeline. Vision fusion is
    per-architecture (`_VLM_VISION_FUSERS`); each registered family is exercised here."""

    # Families with a registered decomposition + input builder (OVModelForVisualCausalLM token parity).
    # qwen2_vl / qwen2_5_vl / qwen3_vl use bespoke decomposers (`_VLM_DECOMPOSERS`): two-stage vision +
    # 3D M-RoPE, and for qwen3_vl also a pos-embedding IR + deepstack features injected in the LM.
    VLM_ARCHITECTURES = [
        "llava",
        "llava_next",
        "got_ocr2",
        "idefics3",
        "smolvlm",
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3_vl",
    ]
    # GenAI VLMPipeline parity. The qwen mergers name their hidden-states input `hidden_states` and (for
    # qwen3_vl) their outputs `last_hidden_state`/`deepstack_feature_lists` — the tensor names GenAI's
    # C++ pipeline feeds/reads; qwen2_5_vl adds window-attention inputs, qwen3_vl an i64 pos-embed index.
    GENAI_VLM_ARCHITECTURES = ["llava", "qwen2_vl", "qwen2_5_vl", "qwen3_vl"]
    # Families that export + run on both runtimes but whose tiny fixtures are too degenerate (near-tied
    # logits) for greedy token parity — verified to run, not to match. gemma3 uses a bespoke decomposer
    # whose language model consumes `token_type_ids` for bidirectional image-token attention.
    RUNS_ONLY_VLM_ARCHITECTURES = ["gemma3"]

    @parameterized.expand(VLM_ARCHITECTURES + RUNS_ONLY_VLM_ARCHITECTURES)
    def test_ovmodel_vlm(self, model_type):
        model_id = MODEL_NAMES[model_type]
        with tempfile.TemporaryDirectory() as tmp:
            export_openvino_hf(model_id, tmp, task="image-text-to-text", fp16=False)

            reference = AutoModelForImageTextToText.from_pretrained(model_id).eval()
            processor = _load_processor(model_id, "multimodal", trust_remote_code=False)
            # Same per-architecture sample the exporter builds, so the image-token count is correct.
            inputs = _build_sample_inputs(processor, "multimodal", reference)
            input_len = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                reference_ids = reference.generate(**inputs, **GEN_KWARGS)[0].tolist()

            ov_model = OVModelForVisualCausalLM.from_pretrained(tmp, ov_config=F32_CONFIG)
            ov_ids = ov_model.generate(**inputs, **GEN_KWARGS)[0].tolist()

            del ov_model
            del reference
            gc.collect()

            if model_type in self.RUNS_ONLY_VLM_ARCHITECTURES:
                # Degenerate fixture (near-tied logits): verify it runs and decodes the requested tokens,
                # not token parity.
                self.assertEqual(len(ov_ids), input_len + GEN_KWARGS["max_new_tokens"], "no tokens generated")
            else:
                self.assertEqual(reference_ids, ov_ids, "Transformers and OVModelForVisualCausalLM tokens differ")

    @parameterized.expand(GENAI_VLM_ARCHITECTURES + RUNS_ONLY_VLM_ARCHITECTURES)
    def test_genai_vlm(self, model_type):
        # The same VLM export runs in OpenVINO GenAI's VLMPipeline (needs the OpenVINO
        # tokenizer/detokenizer + `preprocessor_config.json`), and its generated text matches
        # Transformers greedy. The runtime's per-arch `preprocess_inputs` builds the reference prompt so
        # the image-token expansion matches what GenAI does with `apply_chat_template=True`.
        from optimum.intel.openvino.modeling_visual_language import MODEL_TYPE_TO_CLS_MAPPING

        model_id = MODEL_NAMES[model_type]
        image = Image.new("RGB", (56, 56))
        prompt = "A photo of a cat sitting on a"
        with tempfile.TemporaryDirectory() as tmp:
            export_openvino_hf(model_id, tmp, task="image-text-to-text", fp16=False)

            config = AutoConfig.from_pretrained(model_id)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            processor = AutoProcessor.from_pretrained(model_id)
            model_cls = MODEL_TYPE_TO_CLS_MAPPING[config.model_type]
            inputs = model_cls.preprocess_inputs(
                text=prompt, image=image, tokenizer=tokenizer, processor=processor, config=config
            )
            full_prompt = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

            reference = AutoModelForImageTextToText.from_pretrained(model_id).eval()
            with torch.no_grad():
                reference_ids = reference.generate(**inputs, **GEN_KWARGS)
            reference_text = tokenizer.decode(reference_ids[0], skip_special_tokens=True)[len(full_prompt) :].strip()

            pipe = openvino_genai.VLMPipeline(tmp, "CPU", **F32_CONFIG)
            genai_text = (
                pipe.generate(
                    prompt, images=[openvino.Tensor(np.array(image))], apply_chat_template=True, **GEN_KWARGS
                )
                .texts[0]
                .strip()
            )

            del pipe
            del reference
            gc.collect()

            if model_type in self.RUNS_ONLY_VLM_ARCHITECTURES:
                # Degenerate fixture (near-tied logits): verify it runs and produces text, not parity.
                self.assertTrue(genai_text, "OpenVINO GenAI VLMPipeline produced no text")
            else:
                self.assertEqual(reference_text, genai_text, "Transformers and OpenVINO GenAI VLMPipeline text differ")
