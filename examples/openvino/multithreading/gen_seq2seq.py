import datetime
import threading

from transformers import AutoTokenizer, pipeline

from optimum.intel import OVModelForSeq2SeqLM


model_id = "echarlaix/t5-small-openvino"
model = OVModelForSeq2SeqLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
pipe = pipeline("translation_en_to_fr", model=model, tokenizer=tokenizer)

prompt1 = ["I live in Europe"]
prompt2 = ["What is your name?", "The dog is very happy"]
prompt3 = ["It's a beautiful weather today", "Yes", "Good morning"]
prompts = [prompt1, prompt2, prompt3]

NUM_THREADS = 3

threads = [None] * NUM_THREADS
results = [None] * NUM_THREADS


def print_response(t, p, r):
    print("THREAD", t)
    print("PROMPT:", p)
    for i in range(len(r)):
        print("TRANSLATION", i, ":", r[i]["translation_text"])


def gen_thread(prompt, results, i):
    translations = pipe(prompt)
    results[i] = translations


start = datetime.datetime.now()
for i in range(len(threads)):
    threads[i] = threading.Thread(target=gen_thread, args=(prompts[i], results, i))
    threads[i].start()
nu_trans = 0
for i in range(len(threads)):
    threads[i].join()
    nu_trans += len(results[i])
end = datetime.datetime.now()

for i in range(len(threads)):
    print_response(i, prompts[i], results[i])

print("Generation time [s]", ((end - start).total_seconds()), "translations:", nu_trans)
print("Throughput:", nu_trans / ((end - start).total_seconds()), "translations/s")
