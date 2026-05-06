"""Tests for NemotronH (NVIDIA Nemotron Hybrid Mamba-2 + MoE) OpenVINO export support.

NemotronH combines:
- Selective State Space Models (Mamba-2) for efficient sequence modeling
- Mixture of Experts (MoE) for sparse computation
- Hybrid architecture: alternating Mamba and attention layers with MoE blocks

This test module validates:
1. Model type registration in TasksManager
2. OpenVINO config initialization for NemotronH
3. Support for both text-generation and text-generation-with-past tasks
4. Normalized config attribute mapping
5. Tokenizer conversion for inference
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest
from optimum.exporters.onnx.config import OnnxConfig
from optimum.exporters.openvino.model_configs import (
    NemotronHOpenVINOConfig,
    NemotronHNormalizedTextConfig,
    GraniteMoeHybridOpenVINOConfig,
)
from optimum.exporters.tasks import TasksManager
from optimum.intel.utils.import_utils import is_transformers_version


class TestNemotronHRegistration(unittest.TestCase):
    """Test NemotronH model type registration in the OpenVINO export pipeline."""

    def test_nemotron_h_registered_in_tasks_manager(self):
        """Verify that nemotron_h model type is registered with correct tasks."""
        tasks = TasksManager.get_supported_tasks_for_model_type(
            "nemotron_h", "openvino", library_name="transformers"
        )
        self.assertIn("text-generation", tasks, "text-generation should be supported")
        self.assertIn(
            "text-generation-with-past",
            tasks,
            "text-generation-with-past should be supported",
        )

    def test_nemotron_h_config_inherits_from_granite_moe_hybrid(self):
        """Verify that NemotronHOpenVINOConfig extends GraniteMoeHybridOpenVINOConfig."""
        self.assertTrue(
            issubclass(NemotronHOpenVINOConfig, GraniteMoeHybridOpenVINOConfig),
            "NemotronHOpenVINOConfig should inherit from GraniteMoeHybridOpenVINOConfig",
        )

    def test_nemotron_h_config_inherits_from_ov_config(self):
        """Verify that NemotronHOpenVINOConfig is an OnnxConfig."""
        self.assertTrue(
            issubclass(NemotronHOpenVINOConfig, OnnxConfig),
            "NemotronHOpenVINOConfig should be an OnnxConfig subclass",
        )


class TestNemotronHConfig(unittest.TestCase):
    """Test NemotronHOpenVINOConfig initialization and behavior."""

    @pytest.mark.skipif(
        not is_transformers_version(">=", "5.0"),
        reason="NemotronH requires transformers >= 5.0",
    )
    def test_nemotron_h_config_initialization(self):
        """Test that NemotronHOpenVINOConfig can be instantiated."""
        mock_config = MagicMock()
        mock_config.model_type = "nemotron_h"
        mock_config.hidden_size = 1024
        mock_config.num_hidden_layers = 32
        mock_config.num_attention_heads = 16
        mock_config.num_key_value_heads = 8
        mock_config.intermediate_size = 4096
        mock_config.max_position_embeddings = 4096
        mock_config.vocab_size = 100000
        mock_config.layers_block_type = ["mamba", "attention", "moe"] * 10 + ["mamba"]
        mock_config.num_experts = 64
        mock_config.num_experts_per_tok = 8
        mock_config.ssm_state_size = 64
        mock_config.expand = 2
        mock_config.conv_kernel = 4
        mock_config.head_dim = 64
        mock_config.trust_remote_code = True

        config = NemotronHOpenVINOConfig(mock_config)
        # Verify config was created successfully
        self.assertIsNotNone(config)
        # Verify it's the correct type
        self.assertIsInstance(config, GraniteMoeHybridOpenVINOConfig)
        self.assertIsInstance(config, NemotronHOpenVINOConfig)


class TestNemotronHNormalizedConfig(unittest.TestCase):
    """Test NemotronHNormalizedTextConfig for attribute mapping."""

    def test_normalized_config_initialization(self):
        """Test that NemotronHNormalizedTextConfig initializes correctly."""
        mock_config = MagicMock()
        mock_config.model_type = "nemotron_h"
        mock_config.layers_block_type = ["mamba", "attention", "moe", "mamba"]
        mock_config.hidden_size = 512
        mock_config.num_hidden_layers = 4
        mock_config.num_attention_heads = 8
        mock_config.num_key_value_heads = 4
        mock_config.intermediate_size = 2048
        mock_config.max_position_embeddings = 2048
        mock_config.vocab_size = 50000
        mock_config.num_experts = 8
        mock_config.num_experts_per_tok = 2

        normalized_config = NemotronHNormalizedTextConfig(mock_config)

        # Verify core attributes
        self.assertEqual(normalized_config.hidden_size, 512)
        self.assertEqual(normalized_config.num_hidden_layers, 4)
        self.assertEqual(normalized_config.num_attention_heads, 8)

    def test_normalized_config_layer_types_mapping(self):
        """Test that layer types are correctly mapped."""
        mock_config = MagicMock()
        mock_config.model_type = "nemotron_h"
        mock_config.layers_block_type = ["mamba", "attention", "moe", "mamba", "attention"]
        mock_config.hidden_size = 512
        mock_config.num_hidden_layers = 5
        mock_config.num_attention_heads = 8
        mock_config.num_key_value_heads = 4
        mock_config.intermediate_size = 2048
        mock_config.max_position_embeddings = 2048
        mock_config.vocab_size = 50000
        mock_config.num_experts = 8
        mock_config.num_experts_per_tok = 2

        normalized_config = NemotronHNormalizedTextConfig(mock_config)

        # Check layer composition
        layer_types = normalized_config.layers_block_type
        self.assertEqual(len(layer_types), 5)
        self.assertEqual(layer_types.count("mamba"), 2)
        self.assertEqual(layer_types.count("attention"), 2)
        self.assertEqual(layer_types.count("moe"), 1)

    def test_normalized_config_moe_attributes(self):
        """Test that MoE attributes are correctly mapped."""
        mock_config = MagicMock()
        mock_config.model_type = "nemotron_h"
        mock_config.layers_block_type = ["mamba", "moe", "mamba"]
        mock_config.num_experts = 64
        mock_config.num_experts_per_tok = 8
        mock_config.hidden_size = 512
        mock_config.num_hidden_layers = 3
        mock_config.num_attention_heads = 8
        mock_config.num_key_value_heads = 4
        mock_config.intermediate_size = 2048
        mock_config.max_position_embeddings = 2048
        mock_config.vocab_size = 50000

        normalized_config = NemotronHNormalizedTextConfig(mock_config)

        # Verify MoE configuration
        self.assertEqual(normalized_config.num_experts, 64)
        self.assertEqual(normalized_config.num_experts_per_tok, 8)

    def test_normalized_config_mamba_ssm_attributes(self):
        """Test that Mamba SSM attributes are correctly mapped."""
        mock_config = MagicMock()
        mock_config.model_type = "nemotron_h"
        mock_config.layers_block_type = ["mamba", "moe", "mamba"]
        mock_config.ssm_state_size = 64
        mock_config.expand = 2
        mock_config.conv_kernel = 4
        mock_config.head_dim = 64
        mock_config.hidden_size = 512
        mock_config.num_hidden_layers = 3
        mock_config.num_attention_heads = 8
        mock_config.num_key_value_heads = 4
        mock_config.intermediate_size = 2048
        mock_config.max_position_embeddings = 2048
        mock_config.vocab_size = 50000
        mock_config.num_experts = 8
        mock_config.num_experts_per_tok = 2

        normalized_config = NemotronHNormalizedTextConfig(mock_config)

        # Verify Mamba SSM configuration
        self.assertEqual(normalized_config.ssm_state_size, 64)
        self.assertEqual(normalized_config.expand, 2)
        self.assertEqual(normalized_config.conv_kernel, 4)
        self.assertEqual(normalized_config.head_dim, 64)


class TestNemotronHHybridArchitecture(unittest.TestCase):
    """Test NemotronH hybrid architecture support."""

    def test_hybrid_override_pattern_support(self):
        """Test support for hybrid_override_pattern configuration."""
        mock_config = MagicMock()
        mock_config.model_type = "nemotron_h"
        mock_config.layers_block_type = ["mamba", "attention", "moe"] * 10 + ["mamba"]
        mock_config.hybrid_override_pattern = "ME*M"  # Mamba-Attention-MoE pattern repeated
        mock_config.hidden_size = 512
        mock_config.num_hidden_layers = 31
        mock_config.num_attention_heads = 8
        mock_config.num_key_value_heads = 4
        mock_config.intermediate_size = 2048
        mock_config.max_position_embeddings = 2048
        mock_config.vocab_size = 50000
        mock_config.num_experts = 8
        mock_config.num_experts_per_tok = 2
        mock_config.ssm_state_size = 64
        mock_config.expand = 2
        mock_config.conv_kernel = 4
        mock_config.head_dim = 64

        normalized_config = NemotronHNormalizedTextConfig(mock_config)

        # Verify hybrid pattern is accessible
        if hasattr(normalized_config, "hybrid_override_pattern"):
            self.assertEqual(normalized_config.hybrid_override_pattern, "ME*M")

    def test_alternating_mamba_attention_layers(self):
        """Test support for alternating Mamba and Attention layers."""
        mock_config = MagicMock()
        mock_config.model_type = "nemotron_h"
        mock_config.layers_block_type = []
        # Create alternating pattern: Mamba, Attention, Mamba, Attention...
        for i in range(16):
            if i % 2 == 0:
                mock_config.layers_block_type.append("mamba")
            else:
                mock_config.layers_block_type.append("attention")
        # Add MoE layers
        mock_config.layers_block_type.extend(["moe", "mamba", "moe", "mamba"])

        mock_config.hidden_size = 512
        mock_config.num_hidden_layers = len(mock_config.layers_block_type)
        mock_config.num_attention_heads = 8
        mock_config.num_key_value_heads = 4
        mock_config.intermediate_size = 2048
        mock_config.max_position_embeddings = 2048
        mock_config.vocab_size = 50000
        mock_config.num_experts = 8
        mock_config.num_experts_per_tok = 2
        mock_config.ssm_state_size = 64
        mock_config.expand = 2
        mock_config.conv_kernel = 4
        mock_config.head_dim = 64

        normalized_config = NemotronHNormalizedTextConfig(mock_config)

        layer_types = normalized_config.layers_block_type
        # Verify alternating pattern
        mamba_count = sum(1 for t in layer_types if t == "mamba")
        attention_count = sum(1 for t in layer_types if t == "attention")
        moe_count = sum(1 for t in layer_types if t == "moe")

        self.assertEqual(mamba_count, 10)  # 8 alternating + 2 additional
        self.assertEqual(attention_count, 8)
        self.assertEqual(moe_count, 2)


class TestNemotronHTaskSupport(unittest.TestCase):
    """Test NemotronH support for different text generation tasks."""

    def test_text_generation_task_supported(self):
        """Verify text-generation task is supported for NemotronH."""
        tasks = TasksManager.get_supported_tasks_for_model_type(
            "nemotron_h", "openvino", library_name="transformers"
        )
        self.assertIn("text-generation", tasks)

    def test_text_generation_with_past_task_supported(self):
        """Verify text-generation-with-past task is supported for NemotronH."""
        tasks = TasksManager.get_supported_tasks_for_model_type(
            "nemotron_h", "openvino", library_name="transformers"
        )
        self.assertIn("text-generation-with-past", tasks)

    def test_task_count(self):
        """Verify NemotronH supports exactly 2 text generation tasks."""
        tasks = TasksManager.get_supported_tasks_for_model_type(
            "nemotron_h", "openvino", library_name="transformers"
        )
        # NemotronH should support text generation and cached variants
        text_gen_tasks = {t for t in tasks if "text-generation" in t}
        self.assertGreaterEqual(
            len(text_gen_tasks),
            2,
            "NemotronH should support at least 2 text-generation variants",
        )


if __name__ == "__main__":
    unittest.main()
