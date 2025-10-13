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

import os
import tempfile
import unittest
import subprocess
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import pytest
from transformers import AutoTokenizer

from optimum.intel import OVModelForCausalLM


class WWBIntegrationTest(unittest.TestCase):
    """
    Test Who What Benchmark (WWB) integration for optimum vs genai comparison.
    Tests that WWB can evaluate accuracy metrics between optimum-intel and openvino-genai
    implementations of the same model.
    """

    def setUp(self):
        """Set up test environment."""
        self.test_model_id = "hf-internal-testing/tiny-random-gpt2"
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "wwb_output")
        self.model_path = os.path.join(self.temp_dir, "test_model")

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _prepare_test_model(self) -> str:
        """
        Prepare a test model by converting HF model to OpenVINO format.
        
        Returns:
            str: Path to the prepared OpenVINO model
        """
        # Load and save the model in OpenVINO format
        tokenizer = AutoTokenizer.from_pretrained(self.test_model_id)
        model = OVModelForCausalLM.from_pretrained(self.test_model_id, export=True)
        
        # Save the model and tokenizer
        model.save_pretrained(self.model_path)
        tokenizer.save_pretrained(self.model_path)
        
        return self.model_path

    def _check_wwb_available(self) -> bool:
        """
        Check if WWB (Who What Benchmark) is available in the environment.
        
        Returns:
            bool: True if WWB is available, False otherwise
        """
        try:
            result = subprocess.run(["wwb", "--help"], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _check_openvino_genai_available(self) -> bool:
        """
        Check if openvino-genai is available in the environment.
        
        Returns:
            bool: True if openvino-genai is available, False otherwise
        """
        try:
            import openvino_genai
            return True
        except ImportError:
            return False

    def _run_wwb_optimum_vs_genai(self, model_path: str, 
                                 dataset: str = "hellaswag",
                                 num_samples: int = 5,
                                 device: str = "CPU") -> Dict[str, Any]:
        """
        Run WWB comparison between optimum and genai implementations.
        
        Args:
            model_path: Path to the model to test
            dataset: Dataset to use for evaluation (default: hellaswag)
            num_samples: Number of samples to evaluate (default: 5 for speed)
            device: Device to run evaluation on (default: CPU)
            
        Returns:
            Dict containing the evaluation results
        """
        # Create temporary files for ground truth data
        gt_data_file = os.path.join(self.temp_dir, "reference.csv")
        
        # First, generate ground truth data using optimum (base model)
        base_cmd = [
            "wwb",
            "--base-model", model_path,
            "--gt-data", gt_data_file,
            "--model-type", "text",
            "--dataset", dataset,
            "--num-samples", str(num_samples),
            "--device", device
        ]
        
        try:
            # Generate ground truth
            result = subprocess.run(base_cmd, 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=300,
                                  cwd=self.temp_dir)
            
            if result.returncode != 0:
                raise RuntimeError(f"Base model evaluation failed: {result.stderr}")
                
            # Now run comparison with genai
            target_cmd = [
                "wwb",
                "--target-model", model_path,
                "--gt-data", gt_data_file,
                "--model-type", "text",
                "--genai",
                "--device", device,
                "--output", self.output_dir,
                "--verbose",
                "--dataset", dataset,
                "--num-samples", str(num_samples)
            ]
            
            result = subprocess.run(target_cmd,
                                  capture_output=True,
                                  text=True,
                                  timeout=300,
                                  cwd=self.temp_dir)
            
            if result.returncode != 0:
                raise RuntimeError(f"Target model evaluation failed: {result.stderr}")
                
            return self._parse_wwb_results()
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("WWB evaluation timed out")

    def _parse_wwb_results(self) -> Dict[str, Any]:
        """
        Parse WWB evaluation results from output files.
        
        Returns:
            Dict containing parsed metrics and validation status
        """
        metrics_file = os.path.join(self.output_dir, "metrics.csv")
        metrics_per_question_file = os.path.join(self.output_dir, "metrics_per_question.csv")
        
        results = {
            "metrics_available": False,
            "metrics_per_question_available": False,
            "metrics": {},
            "sample_count": 0,
            "validation_passed": False
        }
        
        # Check if metrics file exists and parse it
        if os.path.exists(metrics_file):
            try:
                metrics_df = pd.read_csv(metrics_file)
                if not metrics_df.empty:
                    results["metrics_available"] = True
                    results["metrics"] = metrics_df.to_dict('records')[0] if len(metrics_df) > 0 else {}
            except Exception as e:
                print(f"Warning: Could not parse metrics file: {e}")
        
        # Check if per-question metrics file exists
        if os.path.exists(metrics_per_question_file):
            try:
                per_question_df = pd.read_csv(metrics_per_question_file)
                if not per_question_df.empty:
                    results["metrics_per_question_available"] = True
                    results["sample_count"] = len(per_question_df)
            except Exception as e:
                print(f"Warning: Could not parse per-question metrics file: {e}")
        
        # Validation: Check if we got meaningful results
        results["validation_passed"] = (
            results["metrics_available"] and 
            results["metrics_per_question_available"] and
            results["sample_count"] > 0
        )
        
        return results

    @pytest.mark.slow
    def test_wwb_optimum_vs_genai_evaluation(self):
        """
        Test WWB evaluation comparing optimum-intel vs openvino-genai.
        
        This test:
        1. Prepares a test model in OpenVINO format
        2. Runs WWB evaluation comparing optimum and genai implementations
        3. Verifies that evaluation metrics were successfully generated
        """
        # Skip test if dependencies are not available
        if not self._check_wwb_available():
            self.skipTest("WWB (Who What Benchmark) is not available in the environment")
            
        if not self._check_openvino_genai_available():
            self.skipTest("openvino-genai is not available in the environment")
        
        # Prepare test model
        model_path = self._prepare_test_model()
        self.assertTrue(os.path.exists(model_path), "Test model preparation failed")
        
        # Run WWB evaluation
        try:
            results = self._run_wwb_optimum_vs_genai(
                model_path=model_path,
                dataset="hellaswag",  # Use a common dataset
                num_samples=3,  # Keep it small for testing
                device="CPU"
            )
            
            # Validate results
            self.assertTrue(results["validation_passed"], 
                          f"WWB evaluation validation failed. Results: {results}")
            
            self.assertTrue(results["metrics_available"], 
                          "Metrics file should be generated and contain data")
            
            self.assertTrue(results["metrics_per_question_available"],
                          "Per-question metrics file should be generated and contain data")
            
            self.assertGreater(results["sample_count"], 0,
                             "Should have evaluated at least one sample")
            
            # Check that we have some basic metrics
            metrics = results["metrics"]
            self.assertIsInstance(metrics, dict, "Metrics should be a dictionary")
            self.assertTrue(len(metrics) > 0, "Metrics dictionary should not be empty")
            
            print(f"WWB evaluation completed successfully. Metrics: {metrics}")
            print(f"Evaluated {results['sample_count']} samples")
            
        except Exception as e:
            self.fail(f"WWB evaluation failed with error: {e}")

    def test_wwb_model_path_validation(self):
        """
        Test that WWB correctly validates model paths.
        """
        if not self._check_wwb_available():
            self.skipTest("WWB is not available in the environment")
            
        # Test with non-existent model path
        non_existent_path = "/path/that/does/not/exist"
        
        with self.assertRaises(RuntimeError):
            self._run_wwb_optimum_vs_genai(
                model_path=non_existent_path,
                num_samples=1
            )

    def test_wwb_output_directory_creation(self):
        """
        Test that WWB creates output directory and files as expected.
        """
        if not self._check_wwb_available():
            self.skipTest("WWB is not available in the environment")
            
        if not self._check_openvino_genai_available():
            self.skipTest("openvino-genai is not available in the environment")
        
        # Prepare test model
        model_path = self._prepare_test_model()
        
        # Ensure output directory doesn't exist initially
        self.assertFalse(os.path.exists(self.output_dir), 
                        "Output directory should not exist initially")
        
        try:
            # Run evaluation
            self._run_wwb_optimum_vs_genai(
                model_path=model_path,
                num_samples=2
            )
            
            # Check that output directory was created
            self.assertTrue(os.path.exists(self.output_dir),
                          "Output directory should be created")
            
            # Check for expected output files
            expected_files = ["metrics.csv", "metrics_per_question.csv"]
            for filename in expected_files:
                file_path = os.path.join(self.output_dir, filename)
                self.assertTrue(os.path.exists(file_path),
                              f"Expected output file {filename} should exist")
                
                # Check that files are not empty
                self.assertGreater(os.path.getsize(file_path), 0,
                                 f"Output file {filename} should not be empty")
                
        except Exception as e:
            # If the test fails due to missing dependencies, skip gracefully
            if "not available" in str(e).lower():
                self.skipTest(f"Test dependencies not available: {e}")
            else:
                raise


if __name__ == "__main__":
    unittest.main()