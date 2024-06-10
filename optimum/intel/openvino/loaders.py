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

import logging
import warnings
from typing import Dict, List, Optional, Union

import torch
from diffusers.utils import _get_model_file

from ..utils.import_utils import is_safetensors_available


if is_safetensors_available():
    import safetensors

import openvino
from huggingface_hub.constants import HF_HUB_OFFLINE, HUGGINGFACE_HUB_CACHE
from openvino.runtime import Type
from openvino.runtime import opset11 as ops
from openvino.runtime.passes import Manager, Matcher, MatcherPass, WrapType
from transformers import PreTrainedTokenizer

from .utils import TEXTUAL_INVERSION_EMBEDDING_KEY, TEXTUAL_INVERSION_NAME, TEXTUAL_INVERSION_NAME_SAFE


try:
    from diffusers.utils import DIFFUSERS_CACHE
except ImportError:
    DIFFUSERS_CACHE = HUGGINGFACE_HUB_CACHE


logger = logging.getLogger(__name__)


class InsertTextEmbedding(MatcherPass):
    r"""
    OpenVINO ngraph transformation for inserting pre-trained texual inversion embedding to text encoder
    """

    def __init__(self, token_ids_and_embeddings):
        MatcherPass.__init__(self)
        self.model_changed = False
        param = WrapType("opset1.Constant")

        def callback(matcher: Matcher) -> bool:
            root = matcher.get_match_root()
            if root.get_friendly_name() == TEXTUAL_INVERSION_EMBEDDING_KEY:
                add_ti = root
                consumers = matcher.get_match_value().get_target_inputs()
                for token_id, embedding in token_ids_and_embeddings:
                    ti_weights = ops.constant(embedding, Type.f32, name=str(token_id))
                    ti_weights_unsqueeze = ops.unsqueeze(ti_weights, axes=0)
                    add_ti = ops.concat(
                        nodes=[add_ti, ti_weights_unsqueeze],
                        axis=0,
                        name=f"{TEXTUAL_INVERSION_EMBEDDING_KEY}.textual_inversion_{token_id}",
                    )

                for consumer in consumers:
                    consumer.replace_source_output(add_ti.output(0))

                # Use new operation for additional matching
                self.register_new_node(add_ti)

            # Root node wasn't replaced or changed
            return False

        self.register_matcher(Matcher(param, "InsertTextEmbedding"), callback)


# Adapted from diffusers.loaders.TextualInversionLoaderMixin
class OVTextualInversionLoaderMixin:
    r"""
    Load textual inversion tokens and embeddings to the tokenizer and text encoder.
    """

    def maybe_convert_prompt(self, prompt: Union[str, List[str]], tokenizer: "PreTrainedTokenizer"):
        r"""
        Processes prompts that include a special token corresponding to a multi-vector textual inversion embedding to
        be replaced with multiple special tokens each corresponding to one of the vectors. If the prompt has no textual
        inversion token or if the textual inversion token is a single vector, the input prompt is returned.

        Parameters:
            prompt (`str` or list of `str`):
                The prompt or prompts to guide the image generation.
            tokenizer (`PreTrainedTokenizer`):
                The tokenizer responsible for encoding the prompt into input tokens.

        Returns:
            `str` or list of `str`: The converted prompt
        """
        if not isinstance(prompt, List):
            prompts = [prompt]
        else:
            prompts = prompt

        prompts = [self._maybe_convert_prompt(p, tokenizer) for p in prompts]

        if not isinstance(prompt, List):
            return prompts[0]

        return prompts

    def _maybe_convert_prompt(self, prompt: str, tokenizer: "PreTrainedTokenizer"):
        r"""
        Maybe convert a prompt into a "multi vector"-compatible prompt. If the prompt includes a token that corresponds
        to a multi-vector textual inversion embedding, this function will process the prompt so that the special token
        is replaced with multiple special tokens each corresponding to one of the vectors. If the prompt has no textual
        inversion token or a textual inversion token that is a single vector, the input prompt is simply returned.

        Parameters:
            prompt (`str`):
                The prompt to guide the image generation.
            tokenizer (`PreTrainedTokenizer`):
                The tokenizer responsible for encoding the prompt into input tokens.

        Returns:
            `str`: The converted prompt
        """
        tokens = tokenizer.tokenize(prompt)
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if token in tokenizer.added_tokens_encoder:
                replacement = token
                i = 1
                while f"{token}_{i}" in tokenizer.added_tokens_encoder:
                    replacement += f" {token}_{i}"
                    i += 1

                prompt = prompt.replace(token, replacement)

        return prompt

    def load_textual_inversion(
        self,
        pretrained_model_name_or_path: Union[str, List[str], Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
        token: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):
        r"""
        Load textual inversion embeddings into the text encoder of [`StableDiffusionPipeline`] (both 🤗 Diffusers and
        Automatic1111 formats are supported).

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike` or `List[str or os.PathLike]` or `Dict` or `List[Dict]`):
                Can be either one of the following or a list of them:

                    - A string, the *model id* (for example `sd-concepts-library/low-poly-hd-logos-icons`) of a
                      pretrained model hosted on the Hub.
                    - A path to a *directory* (for example `./my_text_inversion_directory/`) containing the textual
                      inversion weights.
                    - A path to a *file* (for example `./my_text_inversions.pt`) containing textual inversion weights.
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            token (`str` or `List[str]`, *optional*):
                Override the token to use for the textual inversion weights. If `pretrained_model_name_or_path` is a
                list, then `token` must also be a list of equal length.
            weight_name (`str`, *optional*):
                Name of a custom weight file. This should be used when:

                    - The saved textual inversion file is in 🤗 Diffusers format, but was saved under a specific weight
                      name such as `text_inv.bin`.
                    - The saved textual inversion file is in the Automatic1111 format.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            use_auth_token (Optional[Union[bool, str]], defaults to `None`):
                Deprecated. Please use `token` instead.
            token (Optional[Union[bool, str]], defaults to `None`):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.

        Example:

        To load a textual inversion embedding vector in 🤗 Diffusers format:

        ```py
        from optimum.intel import OVStableDiffusionPipeline

        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = OVStableDiffusionPipeline.from_pretrained(model_id, compile=False)

        pipe.load_textual_inversion("sd-concepts-library/cat-toy")
        pipe.compile()

        prompt = "A <cat-toy> backpack"

        image = pipe(prompt, num_inference_steps=50).images[0]
        image.save("cat-backpack.png")
        ```

        To load a textual inversion embedding vector in Automatic1111 format, make sure to download the vector first
        (for example from [civitAI](https://civitai.com/models/3036?modelVersionId=9857)) and then load the vector
        locally:

        ```py
        from optimum.intel import OVStableDiffusionPipeline

        model_id = "runwayml/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, compile=False)

        pipe.load_textual_inversion("./charturnerv2.pt", token="charturnerv2")
        pipe.compile()

        prompt = "charturnerv2, multiple views of the same character in the same outfit, a character turnaround of a woman wearing a black jacket and red shirt, best quality, intricate details."

        image = pipe(prompt, num_inference_steps=50).images[0]
        image.save("character.png")
        ```
        """

        if not hasattr(self, "tokenizer") or not isinstance(self.tokenizer, PreTrainedTokenizer):
            raise ValueError(
                f"{self.__class__.__name__} requires `self.tokenizer` of type `PreTrainedTokenizer` for calling"
                f" `{self.load_textual_inversion.__name__}`"
            )

        if not hasattr(self, "text_encoder") or not isinstance(self.text_encoder.model, openvino.runtime.Model):
            raise ValueError(
                f"{self.__class__.__name__} requires `self.text_encoder.model` of type `openvino.runtime.Model` for calling"
                f" `{self.load_textual_inversion.__name__}`"
            )

        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
        use_auth_token = kwargs.pop("use_auth_token", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed soon. Please use the `token` argument instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError("You cannot use both `use_auth_token` and `token` arguments at the same time.")
            token = use_auth_token

        if use_safetensors and not is_safetensors_available():
            raise ValueError(
                "`use_safetensors`=True but safetensors is not installed. Please install safetensors with `pip install safetensors"
            )

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = is_safetensors_available()
            allow_pickle = True

        user_agent = {
            "file_type": "text_inversion",
            "framework": "pytorch",
        }

        if not isinstance(pretrained_model_name_or_path, list):
            pretrained_model_name_or_paths = [pretrained_model_name_or_path]
        else:
            pretrained_model_name_or_paths = pretrained_model_name_or_path

        if isinstance(token, str):
            tokens = [token]
        elif token is None:
            tokens = [None] * len(pretrained_model_name_or_paths)
        else:
            tokens = token

        if len(pretrained_model_name_or_paths) != len(tokens):
            raise ValueError(
                f"You have passed a list of models of length {len(pretrained_model_name_or_paths)}, and list of tokens of length {len(tokens)}"
                f"Make sure both lists have the same length."
            )

        valid_tokens = [t for t in tokens if t is not None]
        if len(set(valid_tokens)) < len(valid_tokens):
            raise ValueError(f"You have passed a list of tokens that contains duplicates: {tokens}")

        token_ids_and_embeddings = []

        for pretrained_model_name_or_path, token in zip(pretrained_model_name_or_paths, tokens):
            if not isinstance(pretrained_model_name_or_path, dict):
                # 1. Load textual inversion file
                model_file = None
                # Let's first try to load .safetensors weights
                if (use_safetensors and weight_name is None) or (
                    weight_name is not None and weight_name.endswith(".safetensors")
                ):
                    try:
                        model_file = _get_model_file(
                            pretrained_model_name_or_path,
                            weights_name=weight_name or TEXTUAL_INVERSION_NAME_SAFE,
                            cache_dir=cache_dir,
                            force_download=force_download,
                            resume_download=resume_download,
                            proxies=proxies,
                            local_files_only=local_files_only,
                            use_auth_token=token,  # still uses use_auth_token
                            revision=revision,
                            subfolder=subfolder,
                            user_agent=user_agent,
                        )
                        state_dict = safetensors.torch.load_file(model_file, device="cpu")
                    except Exception as e:
                        if not allow_pickle:
                            raise e

                        model_file = None

                if model_file is None:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path,
                        weights_name=weight_name or TEXTUAL_INVERSION_NAME,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        use_auth_token=token,  # still uses use_auth_token
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                    )
                    state_dict = torch.load(model_file, map_location="cpu")
            else:
                state_dict = pretrained_model_name_or_path

            # 2. Load token and embedding correcly from file
            loaded_token = None
            if isinstance(state_dict, torch.Tensor):
                if token is None:
                    raise ValueError(
                        "You are trying to load a textual inversion embedding that has been saved as a PyTorch tensor. Make sure to pass the name of the corresponding token in this case: `token=...`."
                    )
                embedding = state_dict
            elif len(state_dict) == 1:
                # diffusers
                loaded_token, embedding = next(iter(state_dict.items()))
            elif "string_to_param" in state_dict:
                # A1111
                loaded_token = state_dict["name"]
                embedding = state_dict["string_to_param"]["*"]

            if token is not None and loaded_token != token:
                logger.info(f"The loaded token: {loaded_token} is overwritten by the passed token {token}.")
            else:
                token = loaded_token

            embedding = embedding.detach().cpu().numpy()

            # 3. Make sure we don't mess up the tokenizer or text encoder
            vocab = self.tokenizer.get_vocab()
            if token in vocab:
                raise ValueError(
                    f"Token {token} already in tokenizer vocabulary. Please choose a different token name or remove {token} and embedding from the tokenizer and text encoder."
                )
            elif f"{token}_1" in vocab:
                multi_vector_tokens = [token]
                i = 1
                while f"{token}_{i}" in self.tokenizer.added_tokens_encoder:
                    multi_vector_tokens.append(f"{token}_{i}")
                    i += 1

                raise ValueError(
                    f"Multi-vector Token {multi_vector_tokens} already in tokenizer vocabulary. Please choose a different token name or remove the {multi_vector_tokens} and embedding from the tokenizer and text encoder."
                )
            is_multi_vector = len(embedding.shape) > 1 and embedding.shape[0] > 1
            if is_multi_vector:
                tokens = [token] + [f"{token}_{i}" for i in range(1, embedding.shape[0])]
                embeddings = [e for e in embedding]  # noqa: C416
            else:
                tokens = [token]
                embeddings = [embedding[0]] if len(embedding.shape) > 1 else [embedding]
            # add tokens and get ids
            self.tokenizer.add_tokens(tokens)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            token_ids_and_embeddings += zip(token_ids, embeddings)

            logger.info(f"Loaded textual inversion embedding for {token}.")

        # Insert textual inversion embeddings to text encoder with OpenVINO ngraph transformation
        manager = Manager()
        manager.register_pass(InsertTextEmbedding(token_ids_and_embeddings))
        manager.run_passes(self.text_encoder.model)
