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


import json
import logging
import os
import stat
import warnings
import weakref
from glob import glob
from pathlib import Path
from tempfile import TemporaryDirectory as OrigTemporaryDirectory
from tempfile import mkdtemp
from typing import Tuple, Type, Union

import numpy as np
import torch
from huggingface_hub import model_info
from openvino import Core, Model, properties
from openvino import Type as OVType
from packaging.version import Version
from transformers import AutoTokenizer, CLIPTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.onnx.utils import ParameterFormat, compute_serialized_parameters_size

from optimum.intel.utils.import_utils import is_torch_version


logger = logging.getLogger(__name__)

OV_XML_FILE_NAME = "openvino_model.xml"
OV_ENCODER_NAME = "openvino_encoder_model.xml"
OV_DECODER_NAME = "openvino_decoder_model.xml"
OV_DECODER_WITH_PAST_NAME = "openvino_decoder_with_past_model.xml"
OV_TEXT_EMBEDDINGS_MODEL_NAME = "openvino_text_embeddings_model.xml"
OV_LANGUAGE_MODEL_NAME = "openvino_language_model.xml"
OV_VISION_EMBEDDINGS_MODEL_NAME = "openvino_vision_embeddings_model.xml"
OV_VISION_ENCODER_MODEL_NAME = "openvino_vision_encoder.xml"
ONNX_VISION_ENCODER_MODEL_NAME = "vision_encoder.onnx"
ONNX_PROMPT_ENCODER_MASK_DECODER_MODEL_NAME = "prompt_encoder_mask_decoder.onnx"
OV_PROMPT_ENCODER_MASK_DECODER_MODEL_NAME = "openvino_prompt_encoder_mask_decoder.xml"

OV_TOKENIZER_NAME = "openvino_tokenizer{}.xml"
OV_DETOKENIZER_NAME = "openvino_detokenizer{}.xml"

ONNX_WEIGHTS_NAME = "model.onnx"
ONNX_ENCODER_NAME = "encoder_model.onnx"
ONNX_DECODER_NAME = "decoder_model.onnx"
ONNX_DECODER_WITH_PAST_NAME = "decoder_with_past_model.onnx"

MAX_ONNX_OPSET_2022_2_0 = 10
MAX_ONNX_OPSET = 16
MIN_ONNX_QDQ_OPSET = 13

EXTERNAL_DATA_FORMAT_SIZE_LIMIT = 2 * 1024 * 1024 * 1024

TEXTUAL_INVERSION_EMBEDDING_KEY = "self.text_model.embeddings.token_embedding.weight"
TEXTUAL_INVERSION_EMBEDDING_KEYS = [
    "self.text_model.embeddings.token_embedding.weight",
    "self.model.text_model.embeddings.token_embedding.weight",
]

OV_TO_NP_TYPE = {
    "boolean": np.bool_,
    "i8": np.int8,
    "u8": np.uint8,
    "i16": np.int16,
    "u16": np.uint16,
    "i32": np.int32,
    "u32": np.uint32,
    "i64": np.int64,
    "u64": np.uint64,
    "f16": np.float16,
    "f32": np.float32,
    "f64": np.float64,
}

OV_TO_PT_TYPE = {
    "boolean": torch.bool,
    "i8": torch.int8,
    "u8": torch.uint8,
    "i16": torch.int16,
    "i32": torch.int32,
    "i64": torch.int64,
    "f16": torch.float16,
    "f32": torch.float32,
    "f64": torch.float64,
}

if is_torch_version(">=", "2.4.0"):
    OV_TO_PT_TYPE.update({"u16": torch.uint16, "u32": torch.uint32, "u64": torch.uint64})


STR_TO_OV_TYPE = {
    "boolean": OVType.boolean,
    "f16": OVType.f16,
    "f32": OVType.f32,
    "f64": OVType.f64,
    "i8": OVType.i8,
    "i16": OVType.i16,
    "i32": OVType.i32,
    "i64": OVType.i64,
    "u8": OVType.u8,
    "u16": OVType.u16,
    "u32": OVType.u32,
    "u64": OVType.u64,
    "bf16": OVType.bf16,
}


_HEAD_TO_AUTOMODELS = {
    "feature-extraction": "OVModelForFeatureExtraction",
    "fill-mask": "OVModelForMaskedLM",
    "text-generation": "OVModelForCausalLM",
    "text2text-generation": "OVModelForSeq2SeqLM",
    "text-classification": "OVModelForSequenceClassification",
    "token-classification": "OVModelForTokenClassification",
    "question-answering": "OVModelForQuestionAnswering",
    "image-classification": "OVModelForImageClassification",
    "image-text-to-text": "OVModelForVisualCausalLM",
    "zero-shot-image-classification": "OVModelForZeroShotImageClassification",
    "audio-classification": "OVModelForAudioClassification",
    "stable-diffusion": "OVStableDiffusionPipeline",
    "stable-diffusion-xl": "OVStableDiffusionXLPipeline",
    "stable-diffusion-3": "OVStableDiffusion3Pipeline",
    "sam": "OVSamModel",
    "sana": "OVSanaPipeline",
    "flux": "OVFluxPipeline",
    "flux-fill": "OVFluxFillPipeline",
    "pix2struct": "OVModelForPix2Struct",
    "latent-consistency": "OVLatentConsistencyModelPipeline",
    "open_clip_text": "OVModelOpenCLIPText",
    "open_clip_vision": "OVModelOpenCLIPVisual",
    "open_clip": "OVModelOpenCLIPForZeroShotImageClassification",
    "automatic-speech-recognition": "OVModelForSpeechSeq2Seq",
    "automatic-speech-recognition-with-past": "OVModelForSpeechSeq2Seq",
    "ltx-video": "OVLTXPipeline",
    "text-to-audio": "OVModelForTextToSpeechSeq2Seq",
}

PREDEFINED_CAUSAL_LANGUAGE_DATASETS = {"wikitext2", "c4", "c4-new", "auto"}

PREDEFINED_LANGUAGE_DATASETS = {
    "wikitext2": {"id": "wikitext", "name": "wikitext-2-raw-v1", "split": "train", "streaming": False},
    "c4": {"id": "allenai/c4", "name": "en", "split": "train", "streaming": True},
}

PREDEFINED_SD_DATASETS = {
    "conceptual_captions": {"split": "train", "prompt_column_name": "caption", "streaming": True},
    "laion/220k-GPT4Vision-captions-from-LIVIS": {
        "split": "train",
        "prompt_column_name": "caption",
        "streaming": True,
    },
    "laion/filtered-wit": {"split": "train", "prompt_column_name": "caption", "streaming": True},
}

PREDEFINED_TEXT_IMAGE_ENCODER_DATASETS = {
    "conceptual_captions": {
        "id": "conceptual_captions",
        "split": "train",
        "text_column_name": "caption",
        "image_column_name": "image_url",
        "streaming": True,
    },
}

PREDEFINED_VISUAL_LM_DATASETS = {
    "contextual": {
        "id": "ucla-contextual/contextual_test",
        "split": "test",
        "inputs": {"image_url": "image_url", "instruction": "instruction"},
        "streaming": True,
    }
}

PREDEFINED_SPEECH_TO_TEXT_DATASETS = {
    "librispeech": {
        "id": "openslr/librispeech_asr",
        "name": "clean",
        "split": "validation",
        "streaming": True,
    }
}


NEED_CONVERT_TO_FAST_TOKENIZER: Tuple[Type[PreTrainedTokenizer]] = (CLIPTokenizer,)


def maybe_convert_tokenizer_to_fast(
    hf_tokenizer: PreTrainedTokenizer, tokenizer_path: Path
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    if isinstance(hf_tokenizer, PreTrainedTokenizerFast):
        return hf_tokenizer

    if isinstance(hf_tokenizer, NEED_CONVERT_TO_FAST_TOKENIZER):
        try:
            return AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception:
            return hf_tokenizer

    return hf_tokenizer


def use_external_data_format(num_parameters: int) -> bool:
    """
    Returns whether or not the model requires using external data format for the ONNX export
    Args:
        num_parameters: Number of parameter on the model
    Returns:
        True if model.num_parameters() * size_of(float32) >= 2Gb False otherwise
    """

    return compute_serialized_parameters_size(num_parameters, ParameterFormat.Float) >= EXTERNAL_DATA_FORMAT_SIZE_LIMIT


def _is_timm_ov_dir(model_dir):
    config_file = None
    has_xml = False
    has_bin = False
    if os.path.isdir(model_dir):
        for filename in glob(os.path.join(model_dir, "*")):
            if filename.endswith(".xml"):
                has_xml = True
            if filename.endswith(".bin"):
                has_bin = True
            if filename.endswith("config.json"):
                config_file = filename
    if config_file and has_xml and has_bin:
        with open(config_file) as conf:
            hf_hub_id = json.load(conf).get("hf_hub_id", None)
        if hf_hub_id and model_info(hf_hub_id).library_name == "timm":
            return True
    return False


def _print_compiled_model_properties(compiled_model):
    cur_log_level = logger.getEffectiveLevel()
    logger.setLevel(logging.INFO)
    supported_properties = properties.supported_properties()
    skip_keys = {"SUPPORTED_METRICS", "SUPPORTED_CONFIG_KEYS", supported_properties}
    keys = set(compiled_model.get_property(supported_properties)) - skip_keys
    for k in keys:
        try:
            value = compiled_model.get_property(k)
            if k == properties.device.properties():
                for device_key in value.keys():
                    logger.info(f"  {device_key}:")
                    for k2, value2 in value.get(device_key).items():
                        if k2 not in skip_keys:
                            logger.info(f"    {k2}: {value2}")
            else:
                logger.info(f"  {k}: {value}")
        except Exception:
            logger.error(f"[error] Get property of '{k}' failed")
    try:
        logger.info("EXECUTION_DEVICES:")
        for device in compiled_model.get_property("EXECUTION_DEVICES"):
            logger.info(f"  {device}: {Core().get_property(device, 'FULL_DEVICE_NAME')}")
    except Exception:
        logger.error("[error] Get FULL_DEVICE_NAME failed")
    logger.setLevel(cur_log_level)


def np_to_pt_generators(np_object, device):
    if isinstance(np_object, np.random.RandomState):
        return torch.Generator(device=device).manual_seed(int(np_object.get_state()[1][0]))
    elif isinstance(np_object, np.random.Generator):
        return torch.Generator(device=device).manual_seed(int(np_object.bit_generator.state[1][0]))
    elif isinstance(np_object, list) and isinstance(np_object[0], (np.random.RandomState, np.random.Generator)):
        return [np_to_pt_generators(a, device) for a in np_object]
    elif isinstance(np_object, dict) and isinstance(
        next(iter(np_object.values())), (np.random.RandomState, np.random.Generator)
    ):
        return {k: np_to_pt_generators(v, device) for k, v in np_object.items()}
    else:
        return np_object


def _raise_invalid_batch_size(
    expected_batch_size: int, batch_size: int, num_images_per_prompt: int, guidance_scale: float
):
    current_batch_size = batch_size * num_images_per_prompt * (1 if guidance_scale <= 1 else 2)

    if expected_batch_size != current_batch_size:
        msg = ""
        if guidance_scale is not None and guidance_scale <= 1:
            msg = f"`guidance_scale` was set to {guidance_scale}, static shapes are currently only supported for `guidance_scale` > 1 "

        raise ValueError(
            "The model was statically reshaped and the pipeline inputs do not match the expected shapes. "
            f"The `batch_size`, `num_images_per_prompt` and `guidance_scale` were respectively set to {batch_size}, {num_images_per_prompt} and {guidance_scale}. "
            f"The static model expects an input of size equal to {expected_batch_size} and got the following value instead : {current_batch_size}. "
            f"To fix this, please either provide a different inputs to your model so that `batch_size` * `num_images_per_prompt` * 2 is equal to {expected_batch_size} "
            "or reshape it again accordingly using the `.reshape()` method by setting `batch_size` to -1. " + msg
        )


def get_export_transformers_version(model, config):
    version_str = None

    if isinstance(model, Model):
        if "optimum" in model.rt_info:
            version_str = model.rt_info["optimum"]["transformers_version"].value
    if version_str is None:
        version_str = getattr(config, "transformers_version", "0.0.0")

    version_str = version_str or "0.0.0"

    return Version(version_str)


def model_has_dynamic_inputs(model):
    is_dynamic = False
    for input in model.inputs:
        is_dynamic = input.get_partial_shape().is_dynamic
        if is_dynamic:
            return is_dynamic
    return is_dynamic


# adopted from https://github.com/python/cpython/blob/3.12/Lib/shutil.py for compatibility with python<3.10
def _rmtree(path, ignore_errors=False, onerror=None, *, onexc=None, dir_fd=None):
    """Recursively delete a directory tree.

    If dir_fd is not None, it should be a file descriptor open to a directory;
    path will then be relative to that directory.
    dir_fd may not be implemented on your platform.
    If it is unavailable, using it will raise a NotImplementedError.

    If ignore_errors is set, errors are ignored; otherwise, if onexc or
    onerror is set, it is called to handle the error with arguments (func,
    path, exc_info) where func is platform and implementation dependent;
    path is the argument to that function that caused it to fail; and
    the value of exc_info describes the exception. For onexc it is the
    exception instance, and for onerror it is a tuple as returned by
    sys.exc_info().  If ignore_errors is false and both onexc and
    onerror are None, the exception is reraised.

    onerror is deprecated and only remains for backwards compatibility.
    If both onerror and onexc are set, onerror is ignored and onexc is used.
    """
    _use_fd_functions = (
        {os.open, os.stat, os.unlink, os.rmdir} <= os.supports_dir_fd
        and os.scandir in os.supports_fd
        and os.stat in os.supports_follow_symlinks
    )

    if hasattr(os.stat_result, "st_file_attributes"):

        def _rmtree_islink(path):
            try:
                st = os.lstat(path)
                return stat.S_ISLNK(st.st_mode) or (
                    st.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
                    and st.st_reparse_tag == stat.IO_REPARSE_TAG_MOUNT_POINT
                )
            except OSError:
                return False

    else:

        def _rmtree_islink(path):
            return os.path.islink(path)

    def _rmtree_safe_fd(stack, onexc):
        # Each stack item has four elements:
        # * func: The first operation to perform: os.lstat, os.close or os.rmdir.
        #   Walking a directory starts with an os.lstat() to detect symlinks; in
        #   this case, func is updated before subsequent operations and passed to
        #   onexc() if an error occurs.
        # * dirfd: Open file descriptor, or None if we're processing the top-level
        #   directory given to rmtree() and the user didn't supply dir_fd.
        # * path: Path of file to operate upon. This is passed to onexc() if an
        #   error occurs.
        # * orig_entry: os.DirEntry, or None if we're processing the top-level
        #   directory given to rmtree(). We used the cached stat() of the entry to
        #   save a call to os.lstat() when walking subdirectories.
        func, dirfd, path, orig_entry = stack.pop()
        name = path if orig_entry is None else orig_entry.name
        try:
            if func is os.close:
                os.close(dirfd)
                return
            if func is os.rmdir:
                os.rmdir(name, dir_fd=dirfd)
                return

            # Note: To guard against symlink races, we use the standard
            # lstat()/open()/fstat() trick.
            assert func is os.lstat
            if orig_entry is None:
                orig_st = os.lstat(name, dir_fd=dirfd)
            else:
                orig_st = orig_entry.stat(follow_symlinks=False)

            func = os.open  # For error reporting.
            topfd = os.open(name, os.O_RDONLY | os.O_NONBLOCK, dir_fd=dirfd)

            func = os.path.islink  # For error reporting.
            try:
                if not os.path.samestat(orig_st, os.fstat(topfd)):
                    # Symlinks to directories are forbidden, see GH-46010.
                    raise OSError("Cannot call rmtree on a symbolic link")
                stack.append((os.rmdir, dirfd, path, orig_entry))
            finally:
                stack.append((os.close, topfd, path, orig_entry))

            func = os.scandir  # For error reporting.
            with os.scandir(topfd) as scandir_it:
                entries = list(scandir_it)
            for entry in entries:
                fullname = os.path.join(path, entry.name)
                try:
                    if entry.is_dir(follow_symlinks=False):
                        # Traverse into sub-directory.
                        stack.append((os.lstat, topfd, fullname, entry))
                        continue
                except OSError:
                    pass
                try:
                    os.unlink(entry.name, dir_fd=topfd)
                except OSError as err:
                    onexc(os.unlink, fullname, err)
        except OSError as err:
            err.filename = path
            onexc(func, path, err)

    def _rmtree_unsafe(path, onexc):
        def onerror(err):
            onexc(os.scandir, err.filename, err)

        results = os.walk(path, topdown=False, onerror=onerror, followlinks=hasattr(os, "_walk_symlinks_as_files"))
        for dirpath, dirnames, filenames in results:
            for name in dirnames:
                fullname = os.path.join(dirpath, name)
                try:
                    os.rmdir(fullname)
                except OSError as err:
                    onexc(os.rmdir, fullname, err)
            for name in filenames:
                fullname = os.path.join(dirpath, name)
                try:
                    os.unlink(fullname)
                except OSError as err:
                    onexc(os.unlink, fullname, err)
        try:
            os.rmdir(path)
        except OSError as err:
            onexc(os.rmdir, path, err)

    if ignore_errors:

        def onexc(*args):
            pass

    elif onerror is None and onexc is None:

        def onexc(*args):
            raise

    elif onexc is None:
        if onerror is None:

            def onexc(*args):
                raise

        else:
            # delegate to onerror
            def onexc(*args):
                func, path, exc = args
                if exc is None:
                    exc_info = None, None, None
                else:
                    exc_info = type(exc), exc, exc.__traceback__
                return onerror(func, path, exc_info)

    if _use_fd_functions:
        # While the unsafe rmtree works fine on bytes, the fd based does not.
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        stack = [(os.lstat, dir_fd, path, None)]
        try:
            while stack:
                _rmtree_safe_fd(stack, onexc)
        finally:
            # Close any file descriptors still on the stack.
            while stack:
                func, fd, path, entry = stack.pop()
                if func is not os.close:
                    continue
                try:
                    os.close(fd)
                except OSError as err:
                    onexc(os.close, path, err)
    else:
        if dir_fd is not None:
            raise NotImplementedError("dir_fd unavailable on this platform")
        try:
            if _rmtree_islink(path):
                # symlinks to directories are forbidden, see bug #1669
                raise OSError("Cannot call rmtree on a symbolic link")
        except OSError as err:
            onexc(os.path.islink, path, err)
            # can't continue even if onexc hook returns
            return
        return _rmtree_unsafe(path, onexc)


# copied https://github.com/python/cpython/blob/3.12/Lib/tempfile.py
# to add behaviour that available only for python3.10+ for older python version
class TemporaryDirectory(OrigTemporaryDirectory):
    def __init__(self, suffix=None, prefix=None, dir=None, ignore_cleanup_errors=True, *, delete=True):
        self.name = mkdtemp(suffix, prefix, dir)
        self._ignore_cleanup_errors = ignore_cleanup_errors
        self._delete = delete
        self._finalizer = weakref.finalize(
            self,
            self._cleanup,
            self.name,
            warn_message="Implicitly cleaning up {!r}".format(self),
            ignore_errors=self._ignore_cleanup_errors,
            delete=self._delete,
        )

    @classmethod
    def _cleanup(cls, name, warn_message, ignore_errors=True, delete=True):
        if delete:
            cls._rmtree(name, ignore_errors=ignore_errors)
            warnings.warn(warn_message, ResourceWarning)

    @classmethod
    def _rmtree(cls, name, ignore_errors=True, repeated=False):
        def _dont_follow_symlinks(func, path, *args):
            # Pass follow_symlinks=False, unless not supported on this platform.
            if func in os.supports_follow_symlinks:
                func(path, *args, follow_symlinks=False)
            elif os.name == "nt" or not os.path.islink(path):
                func(path, *args)

        def _resetperms(path):
            try:
                chflags = os.chflags
            except AttributeError:
                pass
            else:
                _dont_follow_symlinks(chflags, path, 0)
            _dont_follow_symlinks(os.chmod, path, 0o700)

        def onexc(func, path, exc):
            if isinstance(exc, PermissionError):
                if repeated and path == name:
                    if ignore_errors:
                        return
                    raise

                try:
                    if path != name:
                        _resetperms(os.path.dirname(path))
                    _resetperms(path)

                    try:
                        os.unlink(path)
                    except IsADirectoryError:
                        cls._rmtree(path, ignore_errors=ignore_errors)
                    except PermissionError:
                        # The PermissionError handler was originally added for
                        # FreeBSD in directories, but it seems that it is raised
                        # on Windows too.
                        # bpo-43153: Calling _rmtree again may
                        # raise NotADirectoryError and mask the PermissionError.
                        # So we must re-raise the current PermissionError if
                        # path is not a directory.
                        if not os.path.isdir(path) or os.path.isjunction(path):
                            if ignore_errors:
                                return
                            raise
                        cls._rmtree(path, ignore_errors=ignore_errors, repeated=(path == name))
                except FileNotFoundError:
                    pass
            elif isinstance(exc, FileNotFoundError):
                pass
            else:
                if not ignore_errors:
                    raise

        _rmtree(name, onexc=onexc, ignore_errors=ignore_errors)

    def cleanup(self):
        if self._finalizer.detach() or os.path.exists(self.name):
            self._rmtree(self.name, ignore_errors=self._ignore_cleanup_errors)


def check_scale_available(model: Union[Model, str, Path]):
    if isinstance(model, Model):
        return model.has_rt_info(["runtime_options", "ACTIVATIONS_SCALE_FACTOR"])
    if not Path(model).exists():
        return False
    import xml.etree.ElementTree as ET

    tree = ET.parse(model)
    root = tree.getroot()
    rt_info = root.find("rt_info")
    if rt_info is None:
        return False
    runtime_options = rt_info.find("runtime_options")
    if runtime_options is None:
        return False
    return runtime_options.find("ACTIVATIONS_SCALE_FACTOR") is not None
