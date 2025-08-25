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
from typing import Dict, List, Optional, Union

import openvino
import torch
from diffusers.loaders.textual_inversion import TextualInversionLoaderMixin, load_textual_inversion_state_dicts
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from openvino import Type
from openvino import opset11 as ops
from openvino.passes import Manager, Matcher, MatcherPass, WrapType
from transformers import PreTrainedTokenizer

from .utils import TEXTUAL_INVERSION_EMBEDDING_KEYS


try:
    from diffusers.utils import DIFFUSERS_CACHE
except ImportError:
    DIFFUSERS_CACHE = HUGGINGFACE_HUB_CACHE


logger = logging.getLogger(__name__)


class InsertTextEmbedding(MatcherPass):
    r"""
    OpenVINO ngraph transformation for inserting pre-trained texual inversion embedding to text encoder
    """

    def __init__(self, tokens_ids, embeddings):
        MatcherPass.__init__(self)

        param = WrapType("opset1.Constant")

        def callback(matcher: Matcher) -> bool:
            root = matcher.get_match_root()
            if root.get_friendly_name() in TEXTUAL_INVERSION_EMBEDDING_KEYS:  # there should be a better way to do this
                add_ti = root
                friendly_name = root.get_friendly_name()
                consumers = matcher.get_match_value().get_target_inputs()
                for token_id, embedding in zip(tokens_ids, embeddings):
                    ti_weights = ops.constant(embedding, Type.f32, name=str(token_id))
                    ti_weights_unsqueeze = ops.unsqueeze(ti_weights, axes=0)
                    add_ti = ops.concat(
                        nodes=[add_ti, ti_weights_unsqueeze],
                        axis=0,
                        name=f"{friendly_name}.textual_inversion_{token_id}",
                    )

                for consumer in consumers:
                    consumer.replace_source_output(add_ti.output(0))

                # Use new operation for additional matching
                self.register_new_node(add_ti)
                return True

            # Root node wasn't replaced or changed
            return False

        self.register_matcher(Matcher(param, "InsertTextEmbedding"), callback)


# Adapted from diffusers.loaders.TextualInversionLoaderMixin
class OVTextualInversionLoaderMixin(TextualInversionLoaderMixin):
    def load_textual_inversion(
        self,
        pretrained_model_name_or_path: Union[str, List[str], Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
        token: Optional[Union[str, List[str]]] = None,
        tokenizer: Optional["PreTrainedTokenizer"] = None,  # noqa: F821
        text_encoder: Optional["openvino.Model"] = None,  # noqa: F821
        **kwargs,
    ):
        if not hasattr(self, "tokenizer"):
            raise ValueError(
                f"{self.__class__.__name__} requires `self.tokenizer` for calling `{self.load_textual_inversion.__name__}`"
            )
        elif not isinstance(self.tokenizer, PreTrainedTokenizer):
            raise ValueError(
                f"{self.__class__.__name__} requires `self.tokenizer` of type `PreTrainedTokenizer` for calling `{self.load_textual_inversion.__name__}`"
            )

        if not hasattr(self, "text_encoder"):
            raise ValueError(
                f"{self.__class__.__name__} requires `self.text_encoder` for calling `{self.load_textual_inversion.__name__}`"
            )
        elif not isinstance(self.text_encoder.model, openvino.Model):
            raise ValueError(
                f"{self.__class__.__name__} requires `self.text_encoder` of type `openvino.Model` for calling `{self.load_textual_inversion.__name__}`"
            )

        # 1. Set correct tokenizer and text encoder
        tokenizer = tokenizer or getattr(self, "tokenizer", None)
        text_encoder = text_encoder or getattr(self, "text_encoder", None)

        # 2. Normalize inputs
        pretrained_model_name_or_paths = (
            [pretrained_model_name_or_path]
            if not isinstance(pretrained_model_name_or_path, list)
            else pretrained_model_name_or_path
        )
        tokens = [token] if not isinstance(token, list) else token
        if tokens[0] is None:
            tokens = tokens * len(pretrained_model_name_or_paths)

        # 3. Check inputs
        self._check_text_inv_inputs(tokenizer, text_encoder, pretrained_model_name_or_paths, tokens)

        # 4. Load state dicts of textual embeddings
        state_dicts = load_textual_inversion_state_dicts(pretrained_model_name_or_paths, **kwargs)

        # 4.1 Handle the special case when state_dict is a tensor that contains n embeddings for n tokens
        if len(tokens) > 1 and len(state_dicts) == 1:
            if isinstance(state_dicts[0], torch.Tensor):
                state_dicts = list(state_dicts[0])
                if len(tokens) != len(state_dicts):
                    raise ValueError(
                        f"You have passed a state_dict contains {len(state_dicts)} embeddings, and list of tokens of length {len(tokens)} "
                        f"Make sure both have the same length."
                    )

        # 4. Retrieve tokens and embeddings
        tokens, embeddings = self._retrieve_tokens_and_embeddings(tokens, state_dicts, tokenizer)

        # 5. Extend tokens and embeddings for multi vector
        tokens, embeddings = self._extend_tokens_and_embeddings(tokens, embeddings, tokenizer)

        # 7.4 add tokens to tokenizer (modified)
        tokenizer.add_tokens(tokens)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Insert textual inversion embeddings to text encoder with OpenVINO ngraph transformation
        manager = Manager()
        manager.register_pass(InsertTextEmbedding(token_ids, embeddings))
        manager.run_passes(text_encoder.model)
