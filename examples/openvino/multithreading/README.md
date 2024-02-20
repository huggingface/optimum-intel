# Execution in multi-threaded environment

## Overview

This example demonstrates how to execute the pipelines from Hugging Face transformers with multi concurency.
A typical scenrio is with multi threaded application without duplicating the model allocation in the host memeory.

By default, the execution of the transformers with OpenVINO Runtime backend is single threaded. Runing the generation process parallel can cause an error
`RuntimeError: Infer Request is busy`.

A simple technic can overcome this limitation using `clone` method on the model or a pipeline. It duplicates the execution object while sharing the OpenVINO compiled model in the host memory. The clone object should not change the model by reshaping, changing precision and recompiling.
The snippet below applies this concept:

```python
pipe = OVStableDiffusionPipeline.from_pretrained(
    MODEL_PATH, ov_config=OV_CONFIG, compile=True
)
def thread(prompt, results):
    pipe_exec = pipe.clone()
    images = pipe_exec(prompt).images
    # Do something with images

T1 = threading.Thread(target=thread, args=("my prompt"))
T1.start()
```
Note that the `clone` operation is quick and is not duplicating the memory usage. It just creates new context for the generating algorithm.

Check the simple examples how it can be applied in practice.

## Preparing python environment
```bash
pip install -r examples/openvino/multithreading/requirement.txt
```

## Text generation

```bash
python examples/openvino/multithreading/gen_text.py
```
## Image generation
```bash
python examples/openvino/multithreading/gen_text.py
```

## Text translation with seq2seq

```bash
python examples/openvino/multithreading/gen_seq2seq.py
```
