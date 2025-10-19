from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    BlipProcessor,
    BlipForConditionalGeneration,
    BitsAndBytesConfig,
    Llama4ForConditionalGeneration,
    Gemma3ForConditionalGeneration,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)
from qwen_vl_utils import process_vision_info
from qwen_omni_utils import process_mm_info

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

from PIL import Image
import torch
from openai import OpenAI
from together import Together
import base64
import io
import os


class Agent:
    def __init__(self, model_name):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if "llava" in model_name:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            self.processor = AutoProcessor.from_pretrained(
                "llava-hf/llava-1.5-7b-hf", trust_remote_code=True, use_fast=False
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                "llava-hf/llava-1.5-7b-hf",
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=(
                    torch.bfloat16 if torch.cuda.is_available() else torch.float32
                ),
                quantization_config=bnb_config,
            ).eval()

        elif "finllava" in model_name.lower():
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            self.processor = AutoProcessor.from_pretrained(
                "TheFinAI/FinLLaVA", trust_remote_code=True
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                "TheFinAI/FinLLaVA",
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=(
                    torch.bfloat16 if torch.cuda.is_available() else torch.float32
                ),
                quantization_config=bnb_config,
            ).eval()

        elif "blip" in model_name:
            self.processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base", trust_remote_code=True
            )
            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=(
                    torch.bfloat16 if torch.cuda.is_available() else torch.float32
                ),
            ).eval()

        elif "qwen" in model_name.lower():

            if "omni" in model_name.lower():
                self.processor = Qwen2_5OmniProcessor.from_pretrained(
                    "Qwen/Qwen2.5-Omni-7B", trust_remote_code=True
                )
                self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2.5-Omni-7B",
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=(
                        torch.bfloat16 if torch.cuda.is_available() else torch.float32
                    ),
                ).eval()

            else:
                self.processor = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen-VL-Max", trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen-VL-Max",
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=(
                        torch.bfloat16 if torch.cuda.is_available() else torch.float32
                    ),
                ).eval()

        elif "deepseek" in model_name:
            model_path = "deepseek-ai/deepseek-vl-7b-chat"

            self.processor = VLChatProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
            self.tokenizer = self.processor.tokenizer

            self.model = MultiModalityCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=(
                    torch.bfloat16 if torch.cuda.is_available() else torch.float32
                ),
            ).eval()

        elif "llama" in model_name.lower():
            self.together_api_key = os.getenv(
                "TOGETHER_API_KEY"
            )

            # model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
            # self.processor = AutoProcessor.from_pretrained(
            #     model_id, trust_remote_code=True
            # )
            # self.model = Llama4ForConditionalGeneration.from_pretrained(
            #     model_id,
            #     trust_remote_code=True,
            #     attn_implementation="flex_attention",
            #     device_map="auto",
            #     torch_dtype=(
            #         torch.bfloat16 if torch.cuda.is_available() else torch.float32
            #     ),
            # ).eval()

        elif "gemma" in model_name:
            self.together_api_key = os.getenv(
                "TOGETHER_API_KEY"
            )

            # if "4b" in model_name:
            #     model_id = "google/gemma-3-4b-it"
            # else:
            #     model_id = "google/gemma-3-27b-it"

            # self.processor = AutoProcessor.from_pretrained(
            #     model_id, trust_remote_code=True
            # )
            # self.model = Gemma3ForConditionalGeneration.from_pretrained(
            #     model_id,
            #     trust_remote_code=True,
            #     device_map="auto",
            #     torch_dtype=(
            #         torch.bfloat16 if torch.cuda.is_available() else torch.float32
            #     ),
            # ).eval()

        elif "gpt-4o" in model_name or "o3-mini" in model_name or "gpt-5" in model_name:
            self.openai_api_key = os.getenv(
                "OPENAI_API_KEY"
            )

        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def draft(self, image_path, local_version=False):
        if local_version:
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path

        prompt = "Convert this financial statement page into semantically correct HTML. Return html and nothing else. Use plain html only, no styling please."

        if "llava" in self.model_name.lower():
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            prompt_str = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = self.processor(
                text=[prompt_str],
                images=[image],
                return_tensors="pt",
            ).to(
                self.device
            )

            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=2048)[:, inputs["input_ids"].shape[-1]:]

            result = self.processor.tokenizer.decode(
                output[0], skip_special_tokens=True
            )
            return result

        elif "blip" in self.model_name:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=2048)
            result = self.processor.tokenizer.decode(
                output[0], skip_special_tokens=True
            )
            return result

        elif "qwen" in self.model_name.lower():

            if "omni" in self.model_name.lower():
                USE_AUDIO_IN_VIDEO = False

                conversation = [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "image": image_path},
                        ],
                    },
                ]

                text = self.processor.apply_chat_template(
                    conversation, add_generation_prompt=True, tokenize=False
                )

                audios, images, videos = process_mm_info(
                    conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO
                )
                inputs = self.processor(
                    text=text,
                    audio=audios,
                    images=images,
                    # videos=videos,
                    return_tensors="pt",
                    padding=True,
                    use_audio_in_video=USE_AUDIO_IN_VIDEO,
                )
                inputs = inputs.to(self.device).to(self.model.dtype)

                with torch.no_grad():
                    text_ids, audio = self.model.generate(
                        **inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO,
                        max_new_tokens=2048
                    )

                if isinstance(text_ids, (list, tuple)):
                    text_ids = text_ids[0]
                result = self.processor.decode(
                    text_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

            else:
                query = self.processor.from_list_format(
                    [
                        {"image": image_path},
                        {"text": prompt},
                    ]
                )

                inputs = self.processor(query, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    output = self.model.generate(**inputs, max_new_tokens=2048)

                result = self.processor.decode(output[0], skip_special_tokens=True)

            return result

        elif "deepseek" in self.model_name.lower():

            conversation = [
                {
                    "role": "User",
                    "content": f"<image_placeholder>{prompt}",
                    "images": [f"{image_path}"],
                },
                {"role": "Assistant", "content": ""},
            ]

            pil_images = load_pil_images(conversation)

            prepare_inputs = self.processor(
                conversations=conversation, images=pil_images, force_batchify=True
            ).to(
                self.model.device,
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )

            with torch.no_grad():

                # run image encoder to get the image embeddings
                inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

                # run the model to get the response
                output = self.model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_new_tokens=2048,
                    do_sample=False,
                    use_cache=True,
                )

            result = self.processor.tokenizer.decode(
                output[0].cpu().tolist(), skip_special_tokens=True
            )

            return result

        # elif "llama" in self.model_name.lower():
        #     messages = [
        #         {
        #             "role": "user",
        #             "content": [
        #                 {"type": "image", "image": image},
        #                 {"type": "text", "text": prompt},
        #             ],
        #         },
        #     ]

        #     inputs = self.processor.apply_chat_template(
        #         messages,
        #         add_generation_prompt=True,
        #         tokenize=True,
        #         return_dict=True,
        #         return_tensors="pt",
        #     ).to(
        #         self.device,
        #         torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        #     )
        #     with torch.no_grad():
        #         output = self.model.generate(**inputs, max_new_tokens=2048)

        #     result = self.processor.tokenizer.decode(
        #         output[0], skip_special_tokens=True
        #     )
        #     return result

        # elif "gemma" in self.model_name:
        #     messages = [
        #         {
        #             "role": "system",
        #             "content": [
        #                 {"type": "text", "text": "You are a helpful assistant."}
        #             ],
        #         },
        #         {
        #             "role": "user",
        #             "content": [
        #                 {"type": "image", "image": image_path},
        #                 {"type": "text", "text": prompt},
        #             ],
        #         },
        #     ]

        #     inputs = self.processor.apply_chat_template(
        #         messages,
        #         add_generation_prompt=True,
        #         tokenize=True,
        #         return_dict=True,
        #         return_tensors="pt",
        #     ).to(
        #         self.model.device,
        #         torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        #     )

        #     with torch.no_grad():
        #         output = self.model.generate(**inputs, max_new_tokens=2048)

        #     decoded = processor.decode(generation, skip_special_tokens=True)

        #     result = self.processor.tokenizer.decode(
        #         output[0], skip_special_tokens=True
        #     )
        #     return result

        elif "gpt-4o" in self.model_name:
            if local_version:
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_bytes = buffered.getvalue()
                b64_image = base64.b64encode(img_bytes).decode("utf-8")
            else:
                b64_image = image_path

            client = OpenAI(
                # This is the default and can be omitted
                api_key=self.openai_api_key,
            )
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_image}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                temperature=0,
                max_tokens=2048
            )
            return response.choices[0].message.content

        elif "gemma" in self.model_name.lower() or "llama" in self.model_name.lower():
            if "Llama-4-Scout-17B-16E-Instruct" in self.model_name:
                model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
            elif "gemma-3-27b-it" in self.model_name:
                model_id = "google/gemma-3-27b-it"
               
            if local_version:
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_bytes = buffered.getvalue()
                b64_image = base64.b64encode(img_bytes).decode("utf-8")
            else:
                b64_image = image_path

            client = Together(
                api_key=self.together_api_key,
            )
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_image}"
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                temperature=0,
                max_tokens=2048,
            )
            return response.choices[0].message.content

        # elif "qwen" in self.model_name.lower():
        #     if local_version:
        #         buffered = io.BytesIO()
        #         image.save(buffered, format="PNG")
        #         img_bytes = buffered.getvalue()
        #         b64_image = base64.b64encode(img_bytes).decode("utf-8")
        #     else:
        #         b64_image = image_path
            
        #     client = OpenAI(
        #         api_key=self.openai_api_key,
        #         base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        #     )
        #     response = client.chat.completions.create(
        #         model="qwen-omni-turbo",
        #         messages=[
        #             {
        #                 "role": "system",
        #                 "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
        #             },
        #             {
        #                 "role": "user",
        #                 "content": [
        #                     {
        #                         "type": "image_url", 
        #                         "image_url": {
        #                             "url": f"data:image/png;base64,{b64_image}"
        #                         },
        #                     },
        #                     {"type": "text", "text": prompt},
        #                 ],
        #             },
        #         ],
        #         modalities=["text", "image"],
        #         stream=True,
        #         stream_options={"include_usage": True},
        #         max_tokens=2048,
        #     )
        #     text = []
        #     for chunk in response:
        #         content = chunk.choices[0].content
        #         if content:
        #             text.append(content)
        #     return "".join(text)
