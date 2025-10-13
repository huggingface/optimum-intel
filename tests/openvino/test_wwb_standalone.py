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
Standalone test for Who What Benchmark (WWB) optimum_vs_genai evaluation.
This test focuses specifically on the WWB CLI functionality without complex optimum.intel dependencies.
"""

import os
import tempfile
import unittest
import subprocess
import shutil
from pathlib import Path

import pandas as pd
import pytest


class WWBStandaloneTest(unittest.TestCase):
    """
    Standalone test for WWB optimum_vs_genai evaluation mode.
    
    Requirements covered:
    - Takes model path as input
    - Runs optimum_vs_genai WWB comparison  
    - Verifies that scores from WWB were evaluated
    """

    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        cls.temp_dir = tempfile.mkdtemp(prefix="wwb_standalone_test_")
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test class."""
        if hasattr(cls, 'temp_dir') and os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def setUp(self):
        """Set up individual test."""
        self.output_dir = os.path.join(self.temp_dir, "wwb_output")
        self.gt_data_file = os.path.join(self.temp_dir, "reference.csv")

    def _check_wwb_available(self):
        """Check if WWB is available."""
        try:
            result = subprocess.run(["wwb", "--help"], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _check_openvino_genai_available(self):
        """Check if openvino-genai is available."""
        try:
            import openvino_genai
            return True
        except ImportError:
            return False

    def _run_wwb_command(self, cmd_args, timeout=60):
        """Run WWB command with error handling."""
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

    def test_wwb_help_command(self):
        """Test that WWB help command works."""
        if not self._check_wwb_available():
            self.skipTest("WWB is not available in the environment")
        
        result = self._run_wwb_command(["wwb", "--help"])
        self.assertEqual(result.returncode, 0, "WWB help command should work")
        self.assertIn("WWB CLI", result.stdout, "Help should contain WWB CLI information")

    def test_wwb_model_path_validation(self):
        """Test WWB model path validation."""
        if not self._check_wwb_available():
            self.skipTest("WWB is not available in the environment")
        
        # Test with obviously invalid path
        invalid_path = "/this/path/definitely/does/not/exist"
        
        cmd = [
            "wwb",
            "--base-model", invalid_path,
            "--gt-data", self.gt_data_file,
            "--model-type", "text",
            "--num-samples", "1"
        ]
        
        result = self._run_wwb_command(cmd, timeout=30)
        
        # Should fail with non-zero exit code for invalid path
        self.assertNotEqual(result.returncode, 0,
                           "WWB should return non-zero exit code for invalid model path")

    @pytest.mark.slow  
    def test_optimum_vs_genai_with_hf_model(self):
        """
        Test optimum_vs_genai evaluation with a HuggingFace model path.
        
        This test demonstrates the main functionality:
        1. Takes model path as input (HF model ID in this case)
        2. Runs optimum_vs_genai comparison
        3. Verifies scores were evaluated
        """
        if not self._check_wwb_available():
            self.skipTest("WWB is not available in the environment")
            
        if not self._check_openvino_genai_available():
            self.skipTest("openvino-genai is not available in the environment")
        
        # Use a tiny HF model for testing
        model_path = "hf-internal-testing/tiny-random-gpt2"
        
        # Step 1: Generate ground truth with optimum (base model)
        print(f"Generating ground truth data with model: {model_path}")
        
        base_cmd = [
            "wwb",
            "--base-model", model_path,
            "--gt-data", self.gt_data_file,
            "--model-type", "text",
            "--dataset", "hellaswag",
            "--num-samples", "2",  # Very small for testing
            "--device", "CPU"
        ]
        
        result = self._run_wwb_command(base_cmd, timeout=180)
        self.assertEqual(result.returncode, 0,
                        f"Base model evaluation failed. stderr: {result.stderr}")
        
        # Verify GT data was created
        self.assertTrue(os.path.exists(self.gt_data_file),
                       "Ground truth data file should be created")
        self.assertGreater(os.path.getsize(self.gt_data_file), 0,
                          "Ground truth data file should not be empty")
        
        # Step 2: Run target evaluation with genai
        print(f"Running genai evaluation with model: {model_path}")
        
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
            "--num-samples", "2"
        ]
        
        result = self._run_wwb_command(target_cmd, timeout=180)
        self.assertEqual(result.returncode, 0,
                        f"Target model evaluation failed. stderr: {result.stderr}")
        
        # Step 3: Verify scores were evaluated
        score_evaluated = self._verify_scores_were_evaluated()
        self.assertTrue(score_evaluated,
                       "Scores from WWB should have been evaluated successfully")
        
        print("✓ WWB optimum_vs_genai evaluation completed successfully")

    def _verify_scores_were_evaluated(self):
        """
        Verify that scores from WWB were evaluated.
        
        This checks the main requirement: "as a result should be if score from wwb was evaluated"
        
        Returns:
            bool: True if scores were successfully evaluated
        """
        # Check that output directory exists
        if not os.path.exists(self.output_dir):
            print(f"Output directory does not exist: {self.output_dir}")
            return False
        
        # Check for required output files
        metrics_file = os.path.join(self.output_dir, "metrics.csv")
        metrics_per_question_file = os.path.join(self.output_dir, "metrics_per_question.csv")
        
        if not os.path.exists(metrics_file):
            print(f"Metrics file does not exist: {metrics_file}")
            return False
            
        if not os.path.exists(metrics_per_question_file):
            print(f"Per-question metrics file does not exist: {metrics_per_question_file}")
            return False
        
        try:
            # Check metrics content
            metrics_df = pd.read_csv(metrics_file)
            per_question_df = pd.read_csv(metrics_per_question_file)
            
            # Verify files contain data
            if metrics_df.empty:
                print("Metrics dataframe is empty")
                return False
                
            if per_question_df.empty:
                print("Per-question metrics dataframe is empty")
                return False
            
            # Check for numerical scores
            numeric_columns = metrics_df.select_dtypes(include=['number']).columns
            if len(numeric_columns) == 0:
                print("No numerical columns found in metrics")
                return False
            
            # Check that we have valid scores (not all NaN)
            has_valid_scores = not metrics_df[numeric_columns].isna().all().all()
            if not has_valid_scores:
                print("All numerical values in metrics are NaN")
                return False
            
            # Print success information
            print(f"✓ Found {len(per_question_df)} evaluated samples")
            print(f"✓ Available metrics: {list(metrics_df.columns)}")
            print(f"✓ Numerical metrics: {list(numeric_columns)}")
            
            return True
            
        except Exception as e:
            print(f"Error parsing results: {e}")
            return False

    def test_score_evaluation_output_structure(self):
        """Test the expected output structure when scores are evaluated."""
        # This test doesn't require WWB, just tests the verification logic
        
        # Create mock output files to test the verification logic
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create a mock metrics.csv file
        mock_metrics = pd.DataFrame({
            'accuracy': [0.75],
            'bleu': [0.68],
            'rouge': [0.72]
        })
        metrics_file = os.path.join(self.output_dir, "metrics.csv")
        mock_metrics.to_csv(metrics_file, index=False)
        
        # Create a mock metrics_per_question.csv file
        mock_per_question = pd.DataFrame({
            'question_id': [1, 2, 3],
            'score': [0.8, 0.7, 0.75],
            'correct': [True, False, True]
        })
        per_question_file = os.path.join(self.output_dir, "metrics_per_question.csv")
        mock_per_question.to_csv(per_question_file, index=False)
        
        # Test verification
        score_evaluated = self._verify_scores_were_evaluated()
        self.assertTrue(score_evaluated, "Mock score evaluation should pass verification")


if __name__ == "__main__":
    unittest.main()