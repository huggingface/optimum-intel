import argparse

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    pipeline,
)


parser = argparse.ArgumentParser()
parser.add_argument("--model_id", default=None, type=str)
parser.add_argument("--dtype", default="bf16", type=str)
parser.add_argument("--jit", default=False, type=bool)
parser.add_argument("--use_ipex", default=False, type=bool)

args = parser.parse_args()

kwargs = {}
if args.dtype == "bf16":
    kwargs["torch_dtype"] = torch.bfloat16
else:
    kwargs["torch_dtype"] = torch.float32
kwargs["use_cache"] = True
kwargs["low_cpu_mem_usage"] = True
kwargs["return_dict"] = True
model_id = args.model_id
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = model.eval()
# model = model.to(memory_format=torch.channels_last)

if kwargs["torch_dtype"] == torch.bfloat16:
    model.to(torch.bfloat16)

input_seq = (
    "Answer the following yes/no question by reasoning step-by-step. Can you write a whole Haiku in a single tweet?"
)
input_length = len(tokenizer(input_seq)["input_ids"])

print("Input sequence is: ")
print(input_seq)

generate_kwargs = {"max_length": 64, "min_length": 8, "do_sample": False, "num_beams": 4, "num_beam_groups": 1}
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
batch_size = generate_kwargs.get("num_beams", 1)

from optimum.exporters import TasksManager


def prepare_jit_inputs(model: PreTrainedModel, task: str, input_length=None, batch_size=None):
    """
    Prepare tuple inputs for jit trace model
    """
    # if task in _TASK_ALIASES.keys():
    #     task = _TASK_ALIASES[task]
    # else:
    #     task = TasksManager.infer_task_from_model(model)
    if hasattr(model.config, "use_cache") and model.config.use_cache:
        task += "-with-past"
    onnx_config_class = TasksManager.get_exporter_config_constructor(
        exporter="onnx",
        model=model,
        task=task,
    )
    dummy_inputs_config = {"sequence_length": 1}
    dummy_inputs_config["batch_size"] = batch_size if batch_size is not None else 1
    onnx_config = onnx_config_class(model.config)
    import pdb

    pdb.set_trace()
    dummy_inputs = onnx_config.generate_dummy_inputs(framework="pt", **dummy_inputs_config)

    if model.config.is_encoder_decoder:
        encoder_inputs = dummy_inputs.pop("input_ids", None)
        if encoder_inputs is None:
            encoder_inputs = dummy_inputs.pop("input_features", None)
        if "seq2seq-lm" in task and input_length is not None:
            encoder_inputs = encoder_inputs.repeat_interleave(input_length, dim=-1)
        if encoder_inputs is not None:
            if hasattr(model, "model"):
                encoder_outputs = model.model.encoder(encoder_inputs, return_dict=False)
            else:
                encoder_outputs = model.encoder(encoder_inputs, return_dict=False)
            dummy_inputs["encoder_outputs"] = encoder_outputs
            encoder_sequence_length = encoder_outputs[0].shape[-2]
            dummy_inputs_config.update({"encoder_sequence_length": encoder_sequence_length})
            dummy_inputs["past_key_values"] = onnx_config.generate_dummy_inputs(framework="pt", **dummy_inputs_config)[
                "past_key_values"
            ]

    if "attention_mask" in dummy_inputs.keys():
        global USE_ATTENTION_MASK
        USE_ATTENTION_MASK = True

    if USE_DICT_INPUTS:
        return dummy_inputs
    else:
        tuple_inputs = ordered_inputs(dummy_inputs, model)
        return tuple_inputs


prepare_jit_inputs(model, "text2text-generation")

# def run_pipeline(generator, num_batches=10):
#     for i in range(num_batches):
#         pre = time.time()
#         out = generator(input_seq, **generate_kwargs)
#         print(f"origin model infer costs {time.time()-pre} seconds")
#         print(out)
#         real_output_token_num = len(tokenizer.batch_encode_plus([out[0]["generated_text"]])["input_ids"][0])
#         print(f"Real output tokens: {real_output_token_num}")


# init_context = []
# if kwargs["torch_dtype"] == torch.bfloat16:
#     init_context.append(torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16))


# init_context.append(torch.inference_mode())
# with ContextManagers(init_context):
#     run_pipeline(generator)
# init_context.pop()


# if args.jit:
#     generator.model.config.return_dict = False
# with ContextManagers(init_context), inference_mode(
#     generator,
#     dtype=kwargs["torch_dtype"],
#     verbose=False,
#     jit=args.jit,
#     use_ipex=args.use_ipex,
#     input_length=input_length,
#     batch_size=batch_size,
# ) as trace_pipe:
#     run_pipeline(trace_pipe)


# generator.model = torch.compile(model)
# print(generator.model)
# generator.model.config.return_dict = True
# init_context.append(torch.inference_mode())
# with ContextManagers(init_context):
#     run_pipeline(generator)
# init_context.pop()
# generator.model = model
