Use Python Launcher to Run Optimum-Intel Optimization with Intel Neural Compressor
=====

Users can use our Python **Launcher** design to run the Python model code as it is with automatic enabling of the Optimum-Intel optimization (by Intel Neural Compressor).

## Quick-Start

Example: Let's run an NLP model using ```run_glue.py``` from HuggingFace Transformers PyTorch [examples](https://github.com/huggingface/transformers/blob/v4.26-release/examples/pytorch/text-classification/run_glue.py).

Pre-requisites:
```bash
pip install transformers torch datasets
```

Generally we run this code with a Python command line like this:
```bash
python run_glue.py --model_name_or_path bert-base-cased --task_name mrpc --do_eval --output_dir result
```

With our **Launcher** design, users can easily apply the Optimum-Intel optimization by simply adding an inline prefix:
```bash
-m optimum.intel.neural_compressor.launcher
```
to the Python command line, and let everything else remain still:
```bash
python -m optimum.intel.neural_compressor.launcher run_glue.py --model_name_or_path bert-base-cased --task_name mrpc --do_eval --output_dir result
```

This will run the code ```run_glue.py``` with the Optimum-Intel optimization automatically enabled, while everything else (e.g. your input arguments for the code) remains still.
