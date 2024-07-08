from optimum.intel import OVStableDiffusionPipeline
from optimum.intel import OVStableDiffusionContrlNetPipeline

from diffusers.training_utils import set_seed

from pathlib import Path
import numpy as np
import torch
from collections import namedtuple
import openvino as ov

from PIL import Image
import sys
from diffusers import UniPCMultistepScheduler


scheduler = UniPCMultistepScheduler.from_config("/home/chentianmeng/workspace/optimum-intel-controlnet/model/stable-diffusion-controlnet-openpose/scheduler/scheduler_config.json")
ov_pipe = OVStableDiffusionContrlNetPipeline.from_pretrained("/home/chentianmeng/workspace/optimum-intel-controlnet/model/stable-diffusion-controlnet-openpose", scheduler=scheduler,compile=False)


np.random.seed(42)

pose = Image.open(Path("/home/chentianmeng/workspace/optimum-intel-controlnet/pose.png"))

prompt = "Dancing Darth Vader, best quality, extremely detailed"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
result = ov_pipe(prompt=prompt, image=pose, num_inference_steps=20, negative_prompt=negative_prompt)

result[0].save("pipeline_0.png")