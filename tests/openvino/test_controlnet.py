import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import OpenposeDetector

import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
import openvino as ov

from collections import namedtuple

import gc
from functools import partial

from typing import Tuple

def visualize_pose_results(
    orig_img: Image.Image,
    skeleton_img: Image.Image,
    left_title: str = "Original image",
    right_title: str = "Pose",
):
    """
    Helper function for pose estimationresults visualization

    Parameters:
       orig_img (Image.Image): original image
       skeleton_img (Image.Image): processed image with body keypoints
       left_title (str): title for the left image
       right_title (str): title for the right image
    Returns:
       fig (matplotlib.pyplot.Figure): matplotlib generated figure contains drawing result
    """
    orig_img = orig_img.resize(skeleton_img.size)
    im_w, im_h = orig_img.size
    is_horizontal = im_h <= im_w
    figsize = (20, 10) if is_horizontal else (10, 20)
    fig, axs = plt.subplots(
        2 if is_horizontal else 1,
        1 if is_horizontal else 2,
        figsize=figsize,
        sharex="all",
        sharey="all",
    )
    fig.patch.set_facecolor("white")
    list_axes = list(axs.flat)
    for a in list_axes:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)
        a.grid(False)
    list_axes[0].imshow(np.array(orig_img))
    list_axes[1].imshow(np.array(skeleton_img))
    list_axes[0].set_title(left_title, fontsize=15)
    list_axes[1].set_title(right_title, fontsize=15)
    fig.subplots_adjust(wspace=0.01 if is_horizontal else 0.00, hspace=0.01 if is_horizontal else 0.1)
    fig.tight_layout()
    return fig



def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()

class OpenPoseOVModel:
    """Helper wrapper for OpenPose model inference"""

    def __init__(self, core, model_path, device="AUTO"):
        self.core = core
        self.model = core.read_model(model_path)
        self.compiled_model = core.compile_model(self.model, device)

    def __call__(self, input_tensor: torch.Tensor):
        """
        inference step

        Parameters:
          input_tensor (torch.Tensor): tensor with prerpcessed input image
        Returns:
           predicted keypoints heatmaps
        """
        h, w = input_tensor.shape[2:]
        input_shape = self.model.input(0).shape
        if h != input_shape[2] or w != input_shape[3]:
            self.reshape_model(h, w)
        results = self.compiled_model(input_tensor)
        return torch.from_numpy(results[self.compiled_model.output(0)]), torch.from_numpy(results[self.compiled_model.output(1)])

    def reshape_model(self, height: int, width: int):
        """
        helper method for reshaping model to fit input data

        Parameters:
          height (int): input tensor height
          width (int): input tensor width
        Returns:
          None
        """
        self.model.reshape({0: [1, 3, height, width]})
        self.compiled_model = self.core.compile_model(self.model)

    def parameters(self):
        Device = namedtuple("Device", ["device"])
        return [Device(torch.device("cpu"))]

controlnet = ControlNetModel.from_pretrained("/home/chentianmeng/workspace/optimum-intel-controlnet/model/control_v11p_sd15_openpose", torch_dtype=torch.float32)
pipe = StableDiffusionControlNetPipeline.from_pretrained("/home/chentianmeng/workspace/optimum-intel-controlnet/model/stable-diffusion-v1-5", controlnet=controlnet)
pose_estimator = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

# controlnet.save_pretrained("/home/chentianmeng/workspace/optimum-intel-controlnet/model/control_v11p_sd15_openpose")
# pipe.save_pretrained("/home/chentianmeng/workspace/optimum-intel-controlnet/model/stable-diffusion-v1-5")
# pose_estimator.save_pretrained("/home/chentianmeng/workspace/optimum-intel-controlnet/model/ControlNet")

example_url = "https://user-images.githubusercontent.com/29454499/224540208-c172c92a-9714-4a7b-857a-b1e54b4d4791.jpg"
img = Image.open(requests.get(example_url, stream=True).raw)

# pose = pose_estimator(img)
# fig = visualize_pose_results(img, pose)
# plt.savefig("torch.png")

OPENPOSE_OV_PATH = Path("/home/chentianmeng/workspace/optimum-intel-controlnet/model/ov_ControlNet/openpose.xml")
if not OPENPOSE_OV_PATH.exists():
    with torch.no_grad():
        ov_model = ov.convert_model(
            pose_estimator.body_estimation.model,
            example_input=torch.zeros([1, 3, 184, 136]),
            input=[[1, 3, 184, 136]],
        )
        ov.save_model(ov_model, OPENPOSE_OV_PATH)
        del ov_model
        cleanup_torchscript_cache()
    print("OpenPose successfully converted to IR")
else:
    print(f"OpenPose will be loaded from {OPENPOSE_OV_PATH}")
    
core = ov.Core() 
ov_openpose = OpenPoseOVModel(core, OPENPOSE_OV_PATH, device="CPU")
pose_estimator.body_estimation.model = ov_openpose  
pose = pose_estimator(img)
pose.save("pose.png")

fig = visualize_pose_results(img, pose)
plt.savefig("ov.png")



inputs = {
    "sample": torch.randn((2, 4, 64, 64)),
    "timestep": torch.tensor(1),
    "encoder_hidden_states": torch.randn((2, 77, 768)),
    "controlnet_cond": torch.randn((2, 3, 512, 512)),
}

# input_info = [(name, ov.PartialShape(inp.shape)) for name, inp in inputs.items()]
input_info = []
for name, inp in inputs.items():
    shape = ov.PartialShape(inp.shape)
    # element_type = dtype_mapping[input_tensor.dtype]
    if len(shape) == 4:
        shape[0] = -1
        shape[2] = -1
        shape[3] = -1
    elif len(shape) == 3:
        shape[0] = -1
    input_info.append((shape))

 
CONTROLNET_OV_PATH = Path("/home/chentianmeng/workspace/optimum-intel-controlnet/model/ov_ControlNet/controlnet-pose.xml")
controlnet.eval()
with torch.no_grad():
    down_block_res_samples, mid_block_res_sample = controlnet(**inputs, return_dict=False)

if not CONTROLNET_OV_PATH.exists():
    with torch.no_grad():
        controlnet.forward = partial(controlnet.forward, return_dict=False)
        ov_model = ov.convert_model(controlnet, example_input=inputs, input=input_info)
        ov.save_model(ov_model, CONTROLNET_OV_PATH)
        del ov_model
        cleanup_torchscript_cache()
    print("ControlNet successfully converted to IR")
else:
    print(f"ControlNet will be loaded from {CONTROLNET_OV_PATH}")

del controlnet




UNET_OV_PATH = Path("/home/chentianmeng/workspace/optimum-intel-controlnet/model/ov_ControlNet/unet_controlnet.xml")

dtype_mapping = {
    torch.float32: ov.Type.f32,
    torch.float64: ov.Type.f64,
    torch.int32: ov.Type.i32,
    torch.int64: ov.Type.i64,
}


class UnetWrapper(torch.nn.Module):
    def __init__(
        self,
        unet,
        sample_dtype=torch.float32,
        timestep_dtype=torch.int64,
        encoder_hidden_states=torch.float32,
        down_block_additional_residuals=torch.float32,
        mid_block_additional_residual=torch.float32,
    ):
        super().__init__()
        self.unet = unet
        self.sample_dtype = sample_dtype
        self.timestep_dtype = timestep_dtype
        self.encoder_hidden_states_dtype = encoder_hidden_states
        self.down_block_additional_residuals_dtype = down_block_additional_residuals
        self.mid_block_additional_residual_dtype = mid_block_additional_residual

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        down_block_additional_residuals: Tuple[torch.Tensor],
        mid_block_additional_residual: torch.Tensor,
    ):
        sample.to(self.sample_dtype)
        timestep.to(self.timestep_dtype)
        encoder_hidden_states.to(self.encoder_hidden_states_dtype)
        down_block_additional_residuals = [res.to(self.down_block_additional_residuals_dtype) for res in down_block_additional_residuals]
        mid_block_additional_residual.to(self.mid_block_additional_residual_dtype)
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
        )


def flattenize_inputs(inputs):
    flatten_inputs = []
    for input_data in inputs:
        if input_data is None:
            continue
        if isinstance(input_data, (list, tuple)):
            flatten_inputs.extend(flattenize_inputs(input_data))
        else:
            flatten_inputs.append(input_data)
    return flatten_inputs


if not UNET_OV_PATH.exists():
    inputs.pop("controlnet_cond", None)
    
    inputs["down_block_additional_residuals"] = down_block_res_samples
    inputs["mid_block_additional_residual"] = mid_block_res_sample

    # for name, input_tensor in inputs.items():
    #     if name == "down_block_additional_residuals":
    #         down_block_additional_residuals = [res for res in down_block_additional_residuals]
    #     shape = ov.PartialShape(input_tensor.shape)
    #     # element_type = dtype_mapping[input_tensor.dtype]
    #     if len(shape) == 4:
    #         shape[0] = -1
    #         shape[2] = -1
    #         shape[3] = -1
    #     input_info.append((shape))
    # print(input_info)
    unet = UnetWrapper(pipe.unet)
    unet.eval()

    with torch.no_grad():
        ov_model = ov.convert_model(unet, example_input=inputs)

    flatten_inputs = flattenize_inputs(inputs.values())
    a = 1

    for input_data, input_tensor in zip(flatten_inputs, ov_model.inputs):
        r_name = input_tensor.get_node().get_friendly_name()
        r_shape = ov.PartialShape(input_data.shape)
        if r_name == "sample":
            r_shape[0] = -1
            r_shape[2] = -1
            r_shape[3] = -1
        elif r_name == "encoder_hidden_states":
            r_shape[0] = -1
        elif r_name == "mid_block_additional_residual":
            r_shape[0] = -1
            r_shape[2] = -1
            r_shape[3] = -1
            
        tn = "down_block_additional_residual."
        if r_name not in ["sample", "timestep", "encoder_hidden_states", "mid_block_additional_residual"]:
            print(r_name)
            r_shape[0] = -1
            r_shape[2] = -1
            r_shape[3] = -1
            n_name = tn + str(a)
            print(n_name)
            if a == 23:
                n_name = "down_block_additional_residual"
            input_tensor.get_node().set_friendly_name(n_name)
            a = a + 2
        print(name, r_shape)
        input_tensor.get_node().set_partial_shape(r_shape)
        input_tensor.get_node().set_element_type(dtype_mapping[input_data.dtype])

    ov_model.validate_nodes_and_infer_types()
    ov.save_model(ov_model, UNET_OV_PATH)
    del ov_model
    cleanup_torchscript_cache()
    del unet
    del pipe.unet
    print("Unet successfully converted to IR")
else:
    del pipe.unet
    print(f"Unet will be loaded from {UNET_OV_PATH}")


TEXT_ENCODER_OV_PATH = Path("/home/chentianmeng/workspace/optimum-intel-controlnet/model/ov_ControlNet/text_encoder.xml")


def convert_encoder(text_encoder: torch.nn.Module, ir_path: Path):
    """
    Convert Text Encoder model to OpenVINO IR.
    Function accepts text encoder model, prepares example inputs for conversion, and convert it to OpenVINO Model
    Parameters:
        text_encoder (torch.nn.Module): text_encoder model
        ir_path (Path): File for storing model
    Returns:
        None
    """
    if not ir_path.exists():
        input_ids = torch.ones((1, 77), dtype=torch.long)
        # switch model to inference mode
        text_encoder.eval()

        # disable gradients calculation for reducing memory consumption
        with torch.no_grad():
            ov_model = ov.convert_model(
                text_encoder,  # model instance
                example_input=input_ids,  # inputs for model tracing
                input=([1, 77],),
            )
            ov.save_model(ov_model, ir_path)
            del ov_model
        cleanup_torchscript_cache()
        print("Text Encoder successfully converted to IR")


if not TEXT_ENCODER_OV_PATH.exists():
    convert_encoder(pipe.text_encoder, TEXT_ENCODER_OV_PATH)
else:
    print(f"Text encoder will be loaded from {TEXT_ENCODER_OV_PATH}")
del pipe.text_encoder



VAE_DECODER_OV_PATH = Path("/home/chentianmeng/workspace/optimum-intel-controlnet/model/ov_ControlNet/vae_decoder.xml")


def convert_vae_decoder(vae: torch.nn.Module, ir_path: Path):
    """
    Convert VAE model to IR format.
    Function accepts pipeline, creates wrapper class for export only necessary for inference part,
    prepares example inputs for convert,
    Parameters:
        vae (torch.nn.Module): VAE model
        ir_path (Path): File for storing model
    Returns:
        None
    """

    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latents):
            return self.vae.decode(latents)

    if not ir_path.exists():
        vae_decoder = VAEDecoderWrapper(vae)
        latent_sample = torch.zeros((1, 4, 64, 64))

        vae_decoder.eval()
        with torch.no_grad():
            ov_model = ov.convert_model(
                vae_decoder,
                example_input=latent_sample,
                input=[
                    (1, 4, 64, 64),
                ],
            )
            
            ov.save_model(ov_model, ir_path)
        del ov_model
        cleanup_torchscript_cache()
        print("VAE decoder successfully converted to IR")


if not VAE_DECODER_OV_PATH.exists():
    convert_vae_decoder(pipe.vae, VAE_DECODER_OV_PATH)
else:
    print(f"VAE decoder will be loaded from {VAE_DECODER_OV_PATH}")




from diffusers import DiffusionPipeline
from transformers import CLIPTokenizer
from typing import Union, List, Optional, Tuple
import cv2


def scale_fit_to_window(dst_width: int, dst_height: int, image_width: int, image_height: int):
    """
    Preprocessing helper function for calculating image size for resize with peserving original aspect ratio
    and fitting image to specific window size

    Parameters:
      dst_width (int): destination window width
      dst_height (int): destination window height
      image_width (int): source image width
      image_height (int): source image height
    Returns:
      result_width (int): calculated width for resize
      result_height (int): calculated height for resize
    """
    im_scale = min(dst_height / image_height, dst_width / image_width)
    return int(im_scale * image_width), int(im_scale * image_height)


def preprocess(image: Image.Image):
    """
    Image preprocessing function. Takes image in PIL.Image format, resizes it to keep aspect ration and fits to model input window 512x512,
    then converts it to np.ndarray and adds padding with zeros on right or bottom side of image (depends from aspect ratio), after that
    converts data to float32 data type and change range of values from [0, 255] to [-1, 1], finally, converts data layout from planar NHWC to NCHW.
    The function returns preprocessed input tensor and padding size, which can be used in postprocessing.

    Parameters:
      image (Image.Image): input image
    Returns:
       image (np.ndarray): preprocessed image tensor
       pad (Tuple[int]): pading size for each dimension for restoring image size in postprocessing
    """
    src_width, src_height = image.size
    dst_width, dst_height = scale_fit_to_window(512, 512, src_width, src_height)
    image = np.array(image.resize((dst_width, dst_height), resample=Image.Resampling.LANCZOS))[None, :]
    pad_width = 512 - dst_width
    pad_height = 512 - dst_height
    pad = ((0, 0), (0, pad_height), (0, pad_width), (0, 0))
    image = np.pad(image, pad, mode="constant")
    image = image.astype(np.float32) / 255.0
    image = image.transpose(0, 3, 1, 2)
    return image, pad


def randn_tensor(
    shape: Union[Tuple, List],
    dtype: Optional[np.dtype] = np.float32,
):
    """
    Helper function for generation random values tensor with given shape and data type

    Parameters:
      shape (Union[Tuple, List]): shape for filling random values
      dtype (np.dtype, *optiona*, np.float32): data type for result
    Returns:
      latents (np.ndarray): tensor with random values with given data type and shape (usually represents noise in latent space)
    """
    latents = np.random.randn(*shape).astype(dtype)

    return latents


class OVContrlNetStableDiffusionPipeline(DiffusionPipeline):
    """
    OpenVINO inference pipeline for Stable Diffusion with ControlNet guidence
    """

    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        scheduler,
        core: ov.Core,
        controlnet: ov.Model,
        text_encoder: ov.Model,
        unet: ov.Model,
        vae_decoder: ov.Model,
        device: str = "AUTO",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.vae_scale_factor = 8
        self.scheduler = scheduler
        self.load_models(core, device, controlnet, text_encoder, unet, vae_decoder)
        self.set_progress_bar_config(disable=True)

    def load_models(
        self,
        core: ov.Core,
        device: str,
        controlnet: ov.Model,
        text_encoder: ov.Model,
        unet: ov.Model,
        vae_decoder: ov.Model,
    ):
        """
        Function for loading models on device using OpenVINO

        Parameters:
          core (Core): OpenVINO runtime Core class instance
          device (str): inference device
          controlnet (Model): OpenVINO Model object represents ControlNet
          text_encoder (Model): OpenVINO Model object represents text encoder
          unet (Model): OpenVINO Model object represents UNet
          vae_decoder (Model): OpenVINO Model object represents vae decoder
        Returns
          None
        """
        self.text_encoder = core.compile_model(text_encoder, device)
        self.text_encoder_out = self.text_encoder.output(0)
        self.register_to_config(controlnet=core.compile_model(controlnet, device))
        self.register_to_config(unet=core.compile_model(unet, device))
        self.unet_out = self.unet.output(0)
        self.vae_decoder = core.compile_model(vae_decoder)
        self.vae_decoder_out = self.vae_decoder.output(0)

    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Image.Image,
        num_inference_steps: int = 10,
        negative_prompt: Union[str, List[str]] = None,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        eta: float = 0.0,
        latents: Optional[np.array] = None,
        output_type: Optional[str] = "pil",
    ):
        """
        Function invoked when calling the pipeline for generation.

        Parameters:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`Image.Image`):
                `Image`, or tensor representing an image batch which will be repainted according to `prompt`.
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            negative_prompt (`str` or `List[str]`):
                negative prompt or prompts for generation
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality. This pipeline requires a value of at least `1`.
            latents (`np.ndarray`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `Image.Image` or `np.array`.
        Returns:
            image ([List[Union[np.ndarray, Image.Image]]): generaited images

        """

        # 1. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # 2. Encode input prompt
        text_embeddings = self._encode_prompt(prompt, negative_prompt=negative_prompt)

        # 3. Preprocess image
        orig_width, orig_height = image.size
        image, pad = preprocess(image)
        height, width = image.shape[-2:]
        if do_classifier_free_guidance:
            image = np.concatenate(([image] * 2))

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = 4
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            text_embeddings.dtype,
            latents,
        )

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand the latents if we are doing classifier free guidance.
                # The latents are expanded 3 times because for pix2pix the guidance\
                # is applied for both the text and the input image.
                latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                result = self.controlnet([latent_model_input, t, text_embeddings, image])
                for _, sample in result.items():
                    print(type(sample))
                    print(sample.shape)

                down_and_mid_blok_samples = [sample * controlnet_conditioning_scale for _, sample in result.items()]

                # predict the noise residual
                noise_pred = self.unet([latent_model_input, t, text_embeddings, *down_and_mid_blok_samples])[self.unet_out]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred[0], noise_pred[1]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(torch.from_numpy(noise_pred), t, torch.from_numpy(latents)).prev_sample.numpy()

                # update progress
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 8. Post-processing
        image = self.decode_latents(latents, pad)

        # 9. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)
            image = [img.resize((orig_width, orig_height), Image.Resampling.LANCZOS) for img in image]
        else:
            image = [cv2.resize(img, (orig_width, orig_width)) for img in image]

        return image

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Union[str, List[str]] = None,
    ):
        """
        Encodes the prompt into text encoder hidden states.

        Parameters:
            prompt (str or list(str)): prompt to be encoded
            num_images_per_prompt (int): number of images that should be generated per prompt
            do_classifier_free_guidance (bool): whether to use classifier free guidance or not
            negative_prompt (str or list(str)): negative prompt to be encoded
        Returns:
            text_embeddings (np.ndarray): text encoder hidden states
        """
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # tokenize input prompts
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        text_input_ids = text_inputs.input_ids

        text_embeddings = self.text_encoder(text_input_ids)[self.text_encoder_out]

        # duplicate text embeddings for each generation per prompt
        if num_images_per_prompt != 1:
            bs_embed, seq_len, _ = text_embeddings.shape
            text_embeddings = np.tile(text_embeddings, (1, num_images_per_prompt, 1))
            text_embeddings = np.reshape(text_embeddings, (bs_embed * num_images_per_prompt, seq_len, -1))

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            max_length = text_input_ids.shape[-1]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            else:
                uncond_tokens = negative_prompt
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="np",
            )

            uncond_embeddings = self.text_encoder(uncond_input.input_ids)[self.text_encoder_out]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = np.tile(uncond_embeddings, (1, num_images_per_prompt, 1))
            uncond_embeddings = np.reshape(uncond_embeddings, (batch_size * num_images_per_prompt, seq_len, -1))

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])

        return text_embeddings

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: np.dtype = np.float32,
        latents: np.ndarray = None,
    ):
        """
        Preparing noise to image generation. If initial latents are not provided, they will be generated randomly,
        then prepared latents scaled by the standard deviation required by the scheduler

        Parameters:
           batch_size (int): input batch size
           num_channels_latents (int): number of channels for noise generation
           height (int): image height
           width (int): image width
           dtype (np.dtype, *optional*, np.float32): dtype for latents generation
           latents (np.ndarray, *optional*, None): initial latent noise tensor, if not provided will be generated
        Returns:
           latents (np.ndarray): scaled initial noise for diffusion
        """
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if latents is None:
            latents = randn_tensor(shape, dtype=dtype)
        else:
            latents = latents

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents: np.array, pad: Tuple[int]):
        """
        Decode predicted image from latent space using VAE Decoder and unpad image result

        Parameters:
           latents (np.ndarray): image encoded in diffusion latent space
           pad (Tuple[int]): each side padding sizes obtained on preprocessing step
        Returns:
           image: decoded by VAE decoder image
        """
        latents = 1 / 0.18215 * latents
        image = self.vae_decoder(latents)[self.vae_decoder_out]
        (_, end_h), (_, end_w) = pad[1:3]
        h, w = image.shape[2:]
        unpad_h = h - end_h
        unpad_w = w - end_w
        image = image[:, :, :unpad_h, :unpad_w]
        image = np.clip(image / 2 + 0.5, 0, 1)
        image = np.transpose(image, (0, 2, 3, 1))
        return image
    

from transformers import CLIPTokenizer
from diffusers import UniPCMultistepScheduler

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)


ov_pipe = OVContrlNetStableDiffusionPipeline(
    tokenizer,
    scheduler,
    core,
    CONTROLNET_OV_PATH,
    TEXT_ENCODER_OV_PATH,
    UNET_OV_PATH,
    VAE_DECODER_OV_PATH,
    device="CPU",
)


np.random.seed(42)

pose = pose_estimator(img)

prompt = "Dancing Darth Vader, best quality, extremely detailed"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
result = ov_pipe(prompt, pose, 20, negative_prompt=negative_prompt)

# cv2.imwrite("result_0.png", result[0])
result[0].save("result_0.png")
