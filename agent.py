from transformers import AutoProcessor, AutoModel,AutoTokenizer,AutoModelForCausalLM,GenerationConfig
import os
import math
import numpy as np
import tempfile
import uuid
import psutil
import torch
import torchvision.transforms as T
# from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
import types
from transformers import GenerationMixin





def build_transform(input_size):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

class QwenVLM:
    def __init__(self, model_path='/hpcwork/qdt11020/weights/Qwen3-VL-8B-Instruct',):
        from transformers import Qwen3VLForConditionalGeneration
        # default: Load the model on the available device(s)
        self.model_path=model_path
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def build_message_list(self,image_list, text_list,interleaved=False):
        content_list = []
        if interleaved:
            # Pair up to min length
            min_len = min(len(image_list), len(text_list))
            for i in range(min_len):
                content_list.append({"type": "image", "image": image_list[i]})
                content_list.append({"type": "text",  "text":  text_list[i]})

            # If image_list is longer → append remaining images
            for img in image_list[min_len:]:
                content_list.append({"type": "image", "image": img})

            # If text_list is longer → append remaining texts
            for txt in text_list[min_len:]:
                content_list.append({"type": "text", "text": txt})
        else:
            for img in image_list:
                content_list.append({"type": "image", "image": img})
            for txt in text_list:
                content_list.append({"type": "text", "text": txt})

        return content_list


    # messages={'image':,'text':,'instruction':,}
    def chat(self, messages, max_new_tokens=1024):

        content_list=self.build_message_list(messages['image'],messages['text'])
        # print("=====content_list=====")
        # print(content_list)
        if messages.get('instruction') is not None:
            content_list.insert(0,{"type": "text", "text": messages['instruction']})
        messages = [
            {
                "role": "user",
                "content": content_list,
            }
        ]

        # Preparation for inference
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        # Preparation for inference
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text

class MedVLM_R1:
    def __init__(self, model_path='/home/homesOnMaster/zzhao/weights/MedVLM-R1/',):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, GenerationConfig
        from qwen_vl_utils import process_vision_info
        # default: Load the model on the available device(s)
        self.proc_fnc=process_vision_info
        self.model_path=model_path
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.temp_generation_config = GenerationConfig(
            max_new_tokens=4096,
            do_sample=False,  
            temperature=1, 
            num_return_sequences=1,
            pad_token_id=151643,
        )


    # messages={'image':,'text':,'instruction':,}
    def chat(self, messages, max_new_tokens=1024):
        question = {
            "image":messages['image'],
            "problem":messages['instruction']
        }

        ### here
        message = [{
            "role": "user",
            "content": [{"type": "image", "image": f"file://{question['image'][0]}"}, {"type": "text","text": question['problem']}]
        }]

        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.proc_fnc(message)
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        generated_ids = self.model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False, generation_config=self.temp_generation_config)

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # print(f'model output: {output_text[0]}')
        return output_text[0]





class InternVLM:
    def __init__(self, model_path='/hpcwork/qdt11020/weights/InternVL3-14B'):
        from decord import VideoReader, cpu
        self.model_path=model_path
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map='auto').eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.generation_config = dict(max_new_tokens=1024, do_sample=True)

    def build_message(self, image_list, text_list,interleaved=False):

        pixel_value_list=[load_image(image, max_num=12).to(torch.bfloat16).to(self.model.device) for image in image_list]
        num_images = len(pixel_value_list)
        pixel_values = torch.cat(pixel_value_list, dim=0)
        num_patches_list = [pixel_values.size(0) for pixel_values in pixel_value_list]
        # print(num_patches_list)

        question = ""
        if interleaved:
            # Pair up to min length
            min_len = min(len(image_list), len(text_list))
            for i in range(min_len):
                question+=f"Image-{i+1}: <image>\n"
                question+=f"{text_list[i]}\n"
            # If image_list is longer → append remaining images
            for img in image_list[min_len:]:
                question+=f"Image-{i+1}: <image>\n"
            # If text_list is longer → append remaining texts
            for txt in text_list[min_len:]:
                question+=f"{txt}\n"
        else:
            for i in range(num_images):
                question+=f"Image-{i+1}: <image>\n"
            for i,txt in enumerate(text_list):
                question+=f"{txt}\n"
        return pixel_values, question, num_patches_list

    # messages={'image':,'text':,'instruction':,}
    def chat(self, messages, max_new_tokens=1024):
        if len(messages['image'])==0:
            question = ""
            for i,txt in enumerate(messages['text']):
                question+=f"{txt}\n"
            question=messages['instruction']+"\n"+question
            response, history = self.model.chat(self.tokenizer, None, question, self.generation_config, history=None, return_history=True)      
        else:
            pixel_values,question,num_patches_list=self.build_message(messages['image'],messages['text'])
            question=messages['instruction']+"\n"+question
            response, history = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config,
                                    num_patches_list=num_patches_list,
                                    history=None, return_history=True)
        # print(f'User: {question}\nAssistant: {response}')


        return response


class PhiVLM:
    def __init__(self, model_path='/hpcwork/qdt11020/weights/Phi-4-multimodal-instruct'):
        from transformers import AutoModelForCausalLM, GenerationConfig
        self.model_path=model_path
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True,
            # if you do not use Ampere or later GPUs, change attention to "eager"
            _attn_implementation='flash_attention_2',
        )
        self.generation_config = GenerationConfig.from_pretrained(model_path)
        # Define prompt structure
        self.user_prompt = '<|user|>'
        self.assistant_prompt = '<|assistant|>'
        self.prompt_suffix = '<|end|>'

    def build_message(self, image_list, text_list,instruction,interleaved=False):

        image_list = [Image.open(img_path) for img_path in image_list]
        prompt=""
        prompt+=self.user_prompt+instruction+"\n"
        if interleaved:
            for i in range(len(image_list)):
                prompt+=f"<|image_{i+1}|>"
                if i < len(text_list):
                    prompt+=' '+text_list[i]
        else:
            for i in range(len(image_list)):
                prompt+=f"<|image_{i+1}|>"
            for txt in text_list:
                prompt+=' '+txt
        prompt+=self.prompt_suffix+self.assistant_prompt

    
        return prompt, image_list

    # messages={'image':,'text':,'instruction':,}
    def chat(self, messages, max_new_tokens=1024):
        
        if len(messages['image'])==0:
            prompt=f"<|system|>{messages['instruction']}<|end|><|user|>{messages['text'][0]}<|end|><|assistant|>"
            inputs = self.processor(text=prompt, return_tensors='pt').to(self.model.device)
        else:
            # Part 1: Image Processing
            prompt,image_list=self.build_message(messages['image'],messages['text'],messages['instruction'])
            # if messages.get('instruction') is not None:
            #     prompt=messages['instruction']+"\n"+prompt
            # print("=====prompt=====")
            # print(prompt)
            # print("=====image_list=====")
            # print(image_list)
            inputs = self.processor(text=prompt, images=image_list, return_tensors='pt').to(self.model.device)

        # Generate response
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            generation_config=self.generation_config,
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response


class DeepSeekVLM:
    def __init__(self, model_path='/hpcwork/qdt11020/weights/DeepSeek-OCR'):
        self.model_path=model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, use_safetensors=True)
        self.model = self.model.eval().cuda().to(torch.bfloat16)


    def concat_images_horizontally(self, image_list, temp_dir='./'):

        if not image_list:
            raise ValueError("image_list can't be empty.")

        if len(image_list) == 1:
            return image_list[0]

        # 1. read all images
        images = [Image.open(p).convert("RGB") for p in image_list]

        # 2. resize images to have the same height
        heights = [im.height for im in images]
        max_h = max(heights)

        resized_images = []
        for im in images:
            if im.height != max_h:
                new_w = int(im.width * max_h / im.height)
                im = im.resize((new_w, max_h), Image.BICUBIC)
            resized_images.append(im)

        # 3. compute total width
        total_w = sum(im.width for im in resized_images)
        new_im = Image.new("RGB", (total_w, max_h), (255, 255, 255))

        # 4. horizontally concatenate images
        x_offset = 0
        for im in resized_images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.width

        # 5. 生成临时文件路径并保存
        if temp_dir is None:
            temp_dir = tempfile.gettempdir()
        os.makedirs(temp_dir, exist_ok=True)

        temp_filename = f"concat_{uuid.uuid4().hex}.jpg"
        temp_path = os.path.join(temp_dir, temp_filename)
        new_im.save(temp_path, format="JPEG")

        return temp_path

    def build_message(self, image_list, text_list):
        image_path=self.concat_images_horizontally(image_list)
        prompt = "<image>\n"+" ".join(text_list)
        return prompt, image_path

    # messages={'image':,'text':,'instruction':,}
    def chat(self, messages,max_new_tokens=1024):
        prompt, temp_image_path = self.build_message(messages['image'],messages['text'])
        if messages.get('instruction') is not None:
            prompt=messages['instruction']+"\n"+prompt

        # print("=====prompt=====")
        # print(prompt)
        # print("=====temp_image_path=====")
        # print(temp_image_path)

        res = self.model.infer(self.tokenizer, prompt=prompt, image_file=temp_image_path, output_path='./',base_size = 1024, image_size = 640, crop_mode=True, save_results = False, test_compress = True, eval_mode=True)
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
        return res


class GemmaVLM:
    def __init__(self, model_path='/hpcwork/qdt11020/weights/gemma-3-12b-it'):
        from transformers import Gemma3ForConditionalGeneration
        self.model_path=model_path
        self.model = Gemma3ForConditionalGeneration.from_pretrained(model_path, device_map="auto").eval()
        self.processor = AutoProcessor.from_pretrained(model_path)


    def build_message(self, message_list):
        prompt=[]
        prompt.append(
            {
                "role":"system",
                "content":[{
                    "type":"text","text":message_list["instruction"]
                }]
            }
            )
        content_list=[]
        image_list=message_list['image']
        text_list=message_list['text']
        for img in image_list:
            content_list.append({"type": "image", "image": img})
        for txt in text_list:
            content_list.append({"type": "text", "text": txt})
        prompt.append(
            {
            "role":"user",
            "content":content_list
            }
            )
        return prompt


    def chat(self, messages, max_new_tokens=1024):
        
        
        prompt = self.build_message(messages)
        # print("=====prompt=====")
        # print(prompt)
        # pdb.set_trace()
        inputs = self.processor.apply_chat_template(
        prompt, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False)
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        # print(decoded)
        return decoded

class VisGLM:
    def __init__(self, model_path='/hpcwork/qdt11020/weights/GLM-4.6V-Flash'):
        from transformers import AutoProcessor, Glm4vForConditionalGeneration
        self.model_path=model_path
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Glm4vForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_path,
            torch_dtype="auto",
            device_map="auto",
        )

    def build_message(self, image_list, text_list):
        content_list = []
        for img in image_list:
            content_list.append({"type": "image", "image": img})
        for txt in text_list:
            content_list.append({"type": "text", "text": txt})


        return content_list

    def chat(self, messages, max_new_tokens=1024):

        content_list=self.build_message(messages['image'],messages['text'])
        if messages.get('instruction') is not None:
            content_list.insert(0,{"type": "text", "text": messages['instruction']})
        messages = [
            {
                "role": "user",
                "content": content_list,
            }
        ]
        
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        inputs.pop("token_type_ids", None)
        generated_ids = self.model.generate(**inputs, max_new_tokens=8192)
        output_text = self.processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

        return output_text