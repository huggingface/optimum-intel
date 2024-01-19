import datetime
import threading

from diffusers import DDIMScheduler

from optimum.intel.openvino import OVStableDiffusionPipeline


MODEL_PATH = "/home/devuser/model_server/demos/python_demos/stable_diffusion/model"
OV_CONFIG = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1"}


pipe = OVStableDiffusionPipeline.from_pretrained(MODEL_PATH, device="CPU", ov_config=OV_CONFIG)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)


# set_seed(10)


prompt1 = [" Zebras in space "]
prompt2 = [" The statue of liberty in New York", " Big Ben in London "]
prompt3 = [" pigs on the grass field", "beach in the storm", "sail yacht on the ocean"]
prompts = [prompt1, prompt2, prompt3]

NUM_THREADS = 3

threads = [None] * NUM_THREADS
results = [None] * NUM_THREADS


def save_response(t, p, r):
    print("THREAD", t)
    print("PROMPT:", p)
    for i in range(len(r)):
        print("IMG:", i)
        r[i].save("img_" + str(t) + "_" + str(i) + ".png", format="PNG")


def gen_thread(prompt, results, i):
    text = prompt
    images = pipe(text).images
    results[i] = images


start = datetime.datetime.now()
for i in range(len(threads)):
    threads[i] = threading.Thread(target=gen_thread, args=(prompts[i], results, i))
    threads[i].start()
nu_img = 0
for i in range(len(threads)):
    threads[i].join()
    nu_img += len(results[i])
end = datetime.datetime.now()

for i in range(len(threads)):
    save_response(i, prompts[i], results[i])

print("Generation time [s]", ((end - start).total_seconds()), "images:", nu_img)
print("Throughput:", nu_img * 60 / ((end - start).total_seconds()), "images/min")
