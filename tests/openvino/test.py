from optimum.intel import OVStableDiffusionPipeline, OVContrlNetStableDiffusionPipeline
from diffusers.training_utils import set_seed

pipe = OVStableDiffusionPipeline.from_pretrained("/home/chentianmeng/workspace/optimum-intel-controlnet/model/stable-diffusion-pokemons-fp32", compile=False)


pipe.reshape(batch_size=1, height=512, width=512, num_images_per_prompt=1)

pipe.compile()

set_seed(42)

prompt = "cartoon bird"
output = pipe(prompt, num_inference_steps=50, output_type="pil")

output.images[0].save("output_0.png")