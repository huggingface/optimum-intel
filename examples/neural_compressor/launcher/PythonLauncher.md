Use Python Launcher CLI to Run Optimum-Intel Optimization with Intel Neural Compressor
=====

Intel Neural Compressor quantization can be applied on your model through the Optimum Intel command-line:

## Quick-Start

Example: Let's run an NLP model using ```run_glue.py``` from HuggingFace Transformers PyTorch [examples](https://github.com/huggingface/transformers/blob/v4.26-release/examples/pytorch/text-classification/run_glue.py).


Generally we run this code with a Python command line like this:

```bash
python run_glue.py --model_name_or_path distilbert-base-uncased-finetuned-sst-2-english --task_name sst2 --do_eval --output_dir result
```

With our Launcher CLI design, users can easily apply the Optimum-Intel optimization by simply using the CLI prefix `optimum-intel-cli inc quantize`:

```bash
optimum-intel-cli inc quantize
```

to the Python command line, and let everything else remain still:

```bash
optimum-intel-cli inc quantize run_glue.py --model_name_or_path distilbert-base-uncased-finetuned-sst-2-english --task_name sst2 --do_eval --output_dir result
```

This will run the code ```run_glue.py``` with the Optimum-Intel optimization automatically enabled, while everything else (e.g. your input arguments for the code) remains still.
