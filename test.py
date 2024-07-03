from optimum.intel import OVStableDiffusionPipeline
# import sys
# full_path = sys.path
# _cpath_ = sys.path[0:-1] 
# sys.path = _cpath_
# print(sys.path)
from optimum.intel import OVStableDiffusionContrlNetPipeline
# sys.path = full_path

from diffusers.training_utils import set_seed

from pathlib import Path
import numpy as np
import torch
from collections import namedtuple
import openvino as ov

from PIL import Image
import sys


# pipe = OVStableDiffusionPipeline.from_pretrained("/home/chentianmeng/workspace/optimum-intel-controlnet/model/stable-diffusion-pokemons-fp32", compile=False)
# print(pipe.scheduler.config)

# pipe.reshape(batch_size=1, height=512, width=512, num_images_per_prompt=1)

# pipe.compile()

# set_seed(42)

# prompt = "cartoon bird"
# output = pipe(prompt, num_inference_steps=50, output_type="pil")

# output.images[0].save("output_0.png")


ov_pipe = OVStableDiffusionContrlNetPipeline.from_pretrained("/home/chentianmeng/workspace/optimum-intel-controlnet/model/stable-diffusion-controlnet-openpose", compile=False)


np.random.seed(42)

pose = Image.open(Path("/home/chentianmeng/workspace/optimum-intel-controlnet/pose.png"))

prompt = "Dancing Darth Vader, best quality, extremely detailed"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
result = ov_pipe(prompt=prompt, image=pose, num_inference_steps=20, negative_prompt=negative_prompt)

# cv2.imwrite("result_0.png", result[0])
result[0].save("pipeline_0.png")