import threading
from datetime import datetime
from transformers import AutoConfig, AutoTokenizer, set_seed
from optimum.intel import OVModelForCausalLM


set_seed(10)
model_id = "togethercomputer/RedPajama-INCITE-Chat-3B-v1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "left"
NUM_THREADS = 3
prompt1 = ["<human>: Question: What is the weather like now? Answer: <bot>"]
prompt2 = ["<human>: Question:  What is Openvino?", "<human>: Question: What the the relativity theory? Answer: <bot>"]
prompt3 = [
    "<human>: Question: Are cats smarter that dogs? Answer: <bot>",
    "<human>: Question: How big is an elephant? Answer: <bot>",
    "<human>: Question: The water in the ocean is much hotter than before? Answer: <bot>",
]
prompts = [prompt1, prompt2, prompt3]

OV_CONFIG = {"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR": "", "NUM_STREAMS": "2"}
model = OVModelForCausalLM.from_pretrained(
    model_id,
    config=AutoConfig.from_pretrained(model_id, trust_remote_code=True),
    ov_config=OV_CONFIG,
    compile=True,
    export=True
)

threads = [None] * NUM_THREADS
results = [None] * NUM_THREADS


def print_response(t, p, r):
    print("THREAD", t)
    print("PROMPT:", p)
    for answer in r:
        print("Answer:")
        print(tokenizer.decode(answer, skip_special_tokens=True))


def gen_thread(prompt, results, i):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    generate_kwargs = {
        "input_ids": inputs.input_ids,
        "max_new_tokens": 200,
        "temperature": 1.0,
        "do_sample": True,
        "top_p": 1.0,
        "top_k": 50,
        "num_beams": 5,
        "repetition_penalty": 1.1,
    }
    start = datetime.now()
    model_exec = model.clone()
    end = datetime.now()
    print("cloning model duration", (end - start).total_seconds() * 1000000, "us")
    outputs = model_exec.generate(**generate_kwargs)
    num_tok = 0
    for x in range(len(prompt)):
        num_tok += outputs[x].numel() - inputs.get("input_ids")[x].numel()
    results[i] = outputs, num_tok


start = datetime.now()
for i in range(len(threads)):
    threads[i] = threading.Thread(target=gen_thread, args=(prompts[i], results, i))
    threads[i].start()

total_tok = 0
for i in range(len(threads)):
    threads[i].join()
    total_tok += results[i][1]
end = datetime.now()

for i in range(len(threads)):
    print_response(i, prompts[i], results[i][0])

print("Generation time [s]", ((end - start).total_seconds()), "tokens:", total_tok)
print("Throughput:", total_tok * 60 / ((end - start).total_seconds()), "tokens/min")
