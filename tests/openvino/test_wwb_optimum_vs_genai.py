# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test for Who What Benchmark (WWB) integration with optimum-intel.
Tests the optimum_vs_genai evaluation mode to compare accuracy between
optimum-intel and openvino-genai implementations.
"""

import os
import tempfile
import unittest
import subprocess
import shutil
from pathlib import Path

import pandas as pd
import pytest
from transformers import AutoTokenizer

try:
    from optimum.intel import OVModelForCausalLM
    OPTIMUM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import optimum.intel: {e}")
    OPTIMUM_AVAILABLE = False
    OVModelForCausalLM = None


class WWBOptimumVsGenaiTest(unittest.TestCase):
    """
    Test WWB (Who What Benchmark) optimum_vs_genai evaluation mode.
    
    This test verifies that WWB can successfully run accuracy comparisons
    between optimum-intel and openvino-genai implementations of the same model.
    """

    @classmethod
    def setUpClass(cls):
        """Set up test class with test model."""
        cls.test_model_id = "hf-internal-testing/tiny-random-gpt2"
        cls.temp_dir = tempfile.mkdtemp(prefix="wwb_test_")
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test class."""
        if hasattr(cls, 'temp_dir') and os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def setUp(self):
        """Set up individual test."""
        self.model_path = os.path.join(self.temp_dir, "test_model")
        self.output_dir = os.path.join(self.temp_dir, "wwb_output")
        self.gt_data_file = os.path.join(self.temp_dir, "reference.csv")

    def _check_dependencies(self):
        """Check if required dependencies are available."""
        # Check optimum.intel availability
        if not OPTIMUM_AVAILABLE:
            return False, False, False
            
        # Check WWB availability
        try:
            result = subprocess.run(["wwb", "--help"], 
                                  capture_output=True, text=True, timeout=5)
            wwb_available = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            wwb_available = False
            
        # Check openvino-genai availability
        try:
            import openvino_genai
            genai_available = True
        except ImportError:
            genai_available = False
            
        return OPTIMUM_AVAILABLE, wwb_available, genai_available

    def _prepare_model(self):
        """Prepare OpenVINO model for testing."""
        if os.path.exists(self.model_path):
            return self.model_path
        
        if not OPTIMUM_AVAILABLE:
            raise RuntimeError("optimum.intel is not available for model preparation")
            
        # Convert HF model to OpenVINO format
        tokenizer = AutoTokenizer.from_pretrained(self.test_model_id)
        model = OVModelForCausalLM.from_pretrained(self.test_model_id, export=True)
        
        # Save model and tokenizer
        model.save_pretrained(self.model_path)
        tokenizer.save_pretrained(self.model_path)
        
        return self.model_path

    def _run_wwb_command(self, cmd_args, timeout=120):
        """Run WWB command with proper error handling."""
        try:
            result = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.temp_dir
            )
            return result
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"WWB command timed out: {' '.join(cmd_args)}")

    @pytest.mark.slow
    def test_optimum_vs_genai_evaluation(self):
        """
        Test optimum_vs_genai WWB evaluation mode.
        
        This test:
        1. Takes a model path as input
        2. Runs WWB optimum_vs_genai comparison
        3. Verifies that scores were evaluated (metrics.csv exists and contains data)
        """
        optimum_available, wwb_available, genai_available = self._check_dependencies()
        
        if not optimum_available:
            self.skipTest("optimum.intel is not available in the environment")
        
        if not wwb_available:
            self.skipTest("WWB is not available in the environment")
        
        if not genai_available:
            self.skipTest("openvino-genai is not available in the environment")
        
        # Prepare model
        model_path = self._prepare_model()
        self.assertTrue(os.path.exists(model_path), "Model preparation failed")
        
        # Step 1: Generate ground truth data with optimum (base model)
        base_cmd = [
            "wwb",
            "--base-model", model_path,
            "--gt-data", self.gt_data_file,
            "--model-type", "text",
            "--dataset", "hellaswag", 
            "--num-samples", "3",  # Small number for testing
            "--device", "CPU"
        ]
        
        result = self._run_wwb_command(base_cmd)
        self.assertEqual(result.returncode, 0, 
                        f"Base model evaluation failed: {result.stderr}")
        
        # Verify ground truth data was generated
        self.assertTrue(os.path.exists(self.gt_data_file),
                       "Ground truth data file should be created")
        self.assertGreater(os.path.getsize(self.gt_data_file), 0,
                          "Ground truth data file should not be empty")
        
        # Step 2: Run comparison with genai (target model)
        target_cmd = [
            "wwb",
            "--target-model", model_path,
            "--gt-data", self.gt_data_file,
            "--model-type", "text",
            "--genai",
            "--device", "CPU",
            "--output", self.output_dir,
            "--verbose",
            "--dataset", "hellaswag",
            "--num-samples", "3"
        ]
        
        result = self._run_wwb_command(target_cmd)
        self.assertEqual(result.returncode, 0,
                        f"Target model evaluation failed: {result.stderr}")
        
        # Step 3: Verify evaluation results
        self._verify_evaluation_results()

    def _verify_evaluation_results(self):
        """Verify that WWB evaluation produced expected results."""
        # Check output directory exists
        self.assertTrue(os.path.exists(self.output_dir),
                       "Output directory should be created")
        
        # Check required output files exist
        metrics_file = os.path.join(self.output_dir, "metrics.csv")
        metrics_per_question_file = os.path.join(self.output_dir, "metrics_per_question.csv")
        
        self.assertTrue(os.path.exists(metrics_file),
                       "metrics.csv should be generated")
        self.assertTrue(os.path.exists(metrics_per_question_file),
                       "metrics_per_question.csv should be generated")
        
        # Verify files are not empty
        self.assertGreater(os.path.getsize(metrics_file), 0,
                          "metrics.csv should contain data")
        self.assertGreater(os.path.getsize(metrics_per_question_file), 0,
                          "metrics_per_question.csv should contain data")
        
        # Verify metrics content
        try:
            metrics_df = pd.read_csv(metrics_file)
            self.assertFalse(metrics_df.empty, "Metrics should contain data")
            
            per_question_df = pd.read_csv(metrics_per_question_file)
            self.assertFalse(per_question_df.empty, "Per-question metrics should contain data")
            self.assertGreater(len(per_question_df), 0, "Should have evaluated some samples")
            
            print(f"✓ WWB evaluation completed successfully")
            print(f"✓ Evaluated {len(per_question_df)} samples")
            print(f"✓ Metrics available: {list(metrics_df.columns)}")
            
        except Exception as e:
            self.fail(f"Failed to parse WWB results: {e}")

    def test_model_path_input_validation(self):
        """Test that WWB properly validates model path input."""
        optimum_available, wwb_available, _ = self._check_dependencies()
        
        if not optimum_available:
            self.skipTest("optimum.intel is not available in the environment")
        
        if not wwb_available:
            self.skipTest("WWB is not available in the environment")
        
        # Test with non-existent model path
        non_existent_path = "/non/existent/model/path"
        
        cmd = [
            "wwb",
            "--base-model", non_existent_path,
            "--gt-data", self.gt_data_file,
            "--model-type", "text",
            "--num-samples", "1"
        ]
        
        result = self._run_wwb_command(cmd, timeout=30)
        self.assertNotEqual(result.returncode, 0,
                           "WWB should fail with non-existent model path")

    def test_score_evaluation_verification(self):
        """
        Test that verifies the core requirement: 
        As a result should be if score from wwb was evaluated
        """
        optimum_available, wwb_available, genai_available = self._check_dependencies()
        
        if not optimum_available:
            self.skipTest("optimum.intel is not available in the environment")
        
        if not wwb_available:
            self.skipTest("WWB is not available in the environment")
            
        if not genai_available:
            self.skipTest("openvino-genai is not available in the environment")
        
        # Prepare and run evaluation
        model_path = self._prepare_model()
        
        # Generate GT data
        base_cmd = [
            "wwb", "--base-model", model_path, "--gt-data", self.gt_data_file,
            "--model-type", "text", "--dataset", "hellaswag", "--num-samples", "2"
        ]
        result = self._run_wwb_command(base_cmd)
        self.assertEqual(result.returncode, 0, "GT generation should succeed")
        
        # Run evaluation
        target_cmd = [
            "wwb", "--target-model", model_path, "--gt-data", self.gt_data_file,
            "--model-type", "text", "--genai", "--output", self.output_dir,
            "--dataset", "hellaswag", "--num-samples", "2"
        ]
        result = self._run_wwb_command(target_cmd)
        self.assertEqual(result.returncode, 0, "Evaluation should succeed")
        
        # Verify score was evaluated (this is the main requirement)
        score_evaluated = self._check_if_score_was_evaluated()
        self.assertTrue(score_evaluated, 
                       "Score from WWB should have been evaluated")

    def _check_if_score_was_evaluated(self):
        """
        Check if score from WWB was evaluated by verifying:
        1. Metrics files exist and contain data
        2. Evaluation metrics are present
        3. Scores are numerical values
        
        Returns:
            bool: True if score was successfully evaluated
        """
        metrics_file = os.path.join(self.output_dir, "metrics.csv")
        
        if not os.path.exists(metrics_file):
            return False
            
        try:
            metrics_df = pd.read_csv(metrics_file)
            
            # Check if dataframe has data
            if metrics_df.empty:
                return False
                
            # Check if we have numerical metrics (scores)
            numeric_columns = metrics_df.select_dtypes(include=['number']).columns
            if len(numeric_columns) == 0:
                return False
                
            # Check if metrics contain valid values (not all NaN)
            has_valid_scores = not metrics_df[numeric_columns].isna().all().all()
            
            return has_valid_scores
            
        except Exception:
            return False


if __name__ == "__main__":
    unittest.main()