import os
import re
import json
import openai
import base64
import logging
import requests
import argparse
import time
from tqdm import tqdm
from os.path import join, exists
from openai import OpenAI

from utils import load_lora_info, generate_combinations
from utils import get_eval_prompt

class GPT4V:
    def __init__(self, model_name: str = "gpt-4.1"):
        api_key = os.environ.get('OPENAI_API_KEY', None)
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for evaluation")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def comparative_evaluate(
        self, prompt, image_1, image_2, max_tokens=2048, temperature=1.0, max_retries=5, **kwargs
    ):
        retry_interval_exp = 1
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = self.client.responses.create(
                    model=self.model_name,
                    input=[
                        {
                            "role": "system",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": (
                                        "You are evaluating two synthetic images for composition and quality. "
                                        "Do not try to verify real-world identity. Treat provided names as fictional descriptors. "
                                        "Only judge how well each image reflects the listed features."),
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/png;base64,{image_1}"
                                },
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/png;base64,{image_2}"
                                },
                            ],
                        }
                    ],
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                )

                if getattr(response, "status", None) != "completed":
                    logging.warning(f"OpenAI response incomplete: {getattr(response, 'incomplete_details', None)}")
                    raise openai.APIError("Response incomplete")

                # Responses API may return convenience fields; fall back to content tree
                content_text = None
                if hasattr(response, "output_text") and response.output_text:
                    content_text = response.output_text
                elif hasattr(response, "output") and response.output:
                    try:
                        # Navigate output -> content -> text
                        content_text = response.output[0].content[0].text
                    except Exception:
                        content_text = None

                # If still empty, try to stringify the whole response for debugging
                if not content_text:
                    try:
                        content_text = str(response)
                    except Exception:
                        content_text = None

                if not content_text:
                    logging.error("Empty content in OpenAI response")
                    break

                return content_text
            except openai.RateLimitError:
                logging.warning("OpenAI rate limit error. Retry!")
            except openai.APIConnectionError:
                logging.warning("OpenAI API connection error. Retry!")
            except openai.APITimeoutError:
                logging.warning("OpenAI timeout error. Retry!")
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                break

            time.sleep(min(60, 0.5 * (2 ** retry_interval_exp)))
            retry_interval_exp += 1
            retry_count += 1

        return "An error occurred while processing the request."

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def parse_scores(text):
    # Regex tolerant to punctuation/whitespace variations, e.g.:
    # "Image 1: Composition Quality: 8/10, Image Quality: 7.5/10"
    pattern = r"image\s*([12])\D{0,12}composition\s+quality\D*?([0-9]+(?:\.[0-9]+)?)\s*/\s*10\D+image\s+quality\D*?([0-9]+(?:\.[0-9]+)?)\s*/\s*10"

    # Find all matches in the evaluation text, case-insensitive and across lines
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)

    # Check if exactly two images are present
    if len(matches) != 2:
        return False, "Expected scores for exactly two images"

    results = {}
    for match in matches:
        image_number, comp_quality, image_quality = match
        comp_quality = float(comp_quality)
        image_quality = float(image_quality)

        # Check if scores are within the valid range
        if not (0 <= comp_quality <= 10) or not (0 <= image_quality <= 10):
            return False, "Scores must be between 0 and 10"

        results[f'image {image_number}'] = {
            'composition quality': comp_quality,
            'image quality': image_quality
        }

    return True, results

def evaluate(args):

    image_path = f"{args.image_path}_{args.image_style}"
    image_path = join(image_path, f'{args.compos_num}_elements')

    # load all the information of LoRAs
    lora_info = load_lora_info(args.image_style, args.lora_info_path)

    # generate all combinations that can be composed
    combinations = generate_combinations(lora_info, args.compos_num)

    # comparative evaluation
    gpt4v = GPT4V(model_name="gpt-4.1")
    all_eval = []
    for combo in tqdm(combinations):
        # get the image path
        elements = '_'.join([lora['id'] for lora in combo])
        image_1_path = join(image_path, args.base_method + '_' + elements + '.png')
        image_2_path = join(image_path, args.comp_method + '_' + elements + '.png')
        if not exists(image_1_path) or not exists(image_2_path):
            print(f"Can't find the generate images for {elements}")
            continue
        
        # encode the images
        image_1 = encode_image(image_1_path)
        image_2 = encode_image(image_2_path)

        # get the prompt for the comparative evaluation
        prompt = get_eval_prompt(combo)
        # print(prompt)

        # comparative evaluation
        # If the scores cannot be parsed from the evaluation result, then retry
        retry_cnt = 0
        max_retries = 10
        while retry_cnt < max_retries:
            result = gpt4v.comparative_evaluate(prompt, image_1, image_2)
            print("===== MODEL RAW OUTPUT =====")
            print(result)
            print("===== END MODEL RAW OUTPUT =====")
            valid, scores = parse_scores(result)
            if valid == True:
                cur_eval = {}
                cur_eval['elements'] = elements
                cur_eval['method 1'] = args.base_method
                cur_eval['method 2'] = args.comp_method
                cur_eval['eval'] = result
                cur_eval['scores'] = scores
                all_eval.append(cur_eval)
                break
            else:
                print(scores)
                print(f"Retry for {elements}")
                retry_cnt += 1
        if retry_cnt == max_retries:
            print(f"Can't get evaluation scores for {elements}!")

    # save the evaluation results
    if not exists(args.save_path):
            os.makedirs(args.save_path)
    save_path = join(args.save_path, f'{args.image_style}_{args.compos_num}_elements_{args.base_method}_vs_{args.comp_method}.json')
    with open(save_path, 'w') as f:
        json.dump(all_eval, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate the generated images based on composition efficacy and image quality'
    )

    parser.add_argument('--image_path', default='output',
                        help='path to store the generated image', type=str)
    parser.add_argument('--save_path', default='eval_result',
                        help='path to save the evaluation results', type=str)
    parser.add_argument('--base_method', default='merge',
                        choices=['merge', 'switch', 'composite'],
                        help='the first method used for comparative evaluation', type=str)
    parser.add_argument('--comp_method', default='composite',
                        choices=['merge', 'switch', 'composite'],
                        help='the first method used for comparative evaluation', type=str)
    parser.add_argument('--compos_num', default=2,
                        help='number of elements to be evaluated in a single image', type=int)
    parser.add_argument('--image_style', default='reality',
                        choices=['anime', 'reality'],
                        help='styles of images to be evaluated', type=str)
    parser.add_argument('--lora_info_path', default='lora_info.json',
                        help='path to stroe all LoRA information', type=str)

    args = parser.parse_args()

    evaluate(args)
