import os
import torch
import argparse
from tqdm import tqdm
from os.path import join, exists
from diffusers import DiffusionPipeline, AutoencoderKL
from diffusers import DPMSolverMultistepScheduler

from utils import load_lora_info
from utils import get_prompt

def main(args):

    # set path based on the image style
    args.save_path = args.save_path + "_" + args.image_style
    args.lora_path = join(args.lora_path, args.image_style)

    # load all the information of LoRAs
    lora_info = load_lora_info(args.image_style, args.lora_info_path)

    # set base model based on the image style
    if args.image_style == 'anime':
        model_name = 'gsdf/Counterfeit-V2.5'
    else:
        model_name = 'SG161222/Realistic_Vision_V5.1_noVAE'

    pipeline = DiffusionPipeline.from_pretrained(
        model_name,
        custom_pipeline="MingZhong/StableDiffusionPipeline-with-LoRA-C",
        # torch_dtype=torch.float16,
        use_safetensors=True
    ).to("cuda")

    # set vae
    if args.image_style == "reality":
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            # torch_dtype=torch.float16
        ).to("cuda")
        pipeline.vae = vae

    # set scheduler
    schedule_config = dict(pipeline.scheduler.config)
    schedule_config["algorithm_type"] = "dpmsolver++"
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(schedule_config)

    # initialize LoRAs
    for element in list(lora_info.keys()):
        for lora in lora_info[element]:
            pipeline.load_lora_weights(
                args.lora_path,
                weight_name=lora['id'] + '.safetensors',
                adapter_name=lora['id']
            )

    # prompt initialization
    init_prompt, negative_prompt = get_prompt(args.image_style)

    # generate images for each individual LoRA
    all_loras = [lora for element in lora_info.values() for lora in element]
    for lora in tqdm(all_loras):
        lora_id = lora['id']
        triggers = lora['trigger']
        prompt = init_prompt + ', ' + ', '.join(triggers)

        pipeline.set_adapters([lora_id])

        # reset seed for each LoRA to make outputs comparable
        generator = torch.manual_seed(args.seed)

        image = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.denoise_steps,
            guidance_scale=args.cfg_scale,
            generator=generator,
            cross_attention_kwargs={"scale": args.lora_scale}
        ).images[0]

        save_path = join(args.save_path, 'single_lora')
        if not exists(save_path):
            os.makedirs(save_path)
        file_name = f"single_{lora_id}.png"
        image.save(join(save_path, file_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate images using a single LoRA at a time'
    )

    # Arguments for composing LoRAs
    parser.add_argument('--save_path', default='output',
                        help='path to save the generated image', type=str)
    parser.add_argument('--lora_path', default='models/lora',
                        help='path to store all LoRA models', type=str)
    parser.add_argument('--lora_info_path', default='lora_info.json',
                        help='path to store all LoRA information', type=str)
    parser.add_argument('--lora_scale', default=0.8,
                        help='scale of each LoRA when generating images', type=float)

    # Arguments for generating images
    parser.add_argument('--height', default=512,
                        help='height of the generated images', type=int)
    parser.add_argument('--width', default=512,
                        help='width of the generated images', type=int)
    parser.add_argument('--denoise_steps', default=200,
                        help='number of the denoising steps', type=int)
    parser.add_argument('--cfg_scale', default=10,
                        help='scale for classifier-free guidance', type=float)
    parser.add_argument('--seed', default=111,
                        help='seed for generating images', type=int)
    parser.add_argument('--image_style', default='anime',
                        choices=['anime', 'reality'],
                        help='styles of the generated images', type=str)

    args = parser.parse_args()

    main(args)
