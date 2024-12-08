import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
from torch.nn.utils.rnn import pad_sequence

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    
    model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)
    model.to(device='cuda')    
    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    image_processor = ImagePreprocess(image_processor, data_args)
    file =  open(os.path.expanduser(args.question_file), "r")
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    batch_size = args.batch_size

    for batch_start in tqdm(range(0, len(questions), batch_size)):
        batch = questions[batch_start:batch_start+batch_size]

        input_ids_list = []
        image_tensors_list = []
        image_sizes_list = []
        idx_list = []
        prompt_list = []

        for line in batch:
            idx = line["question_id"]
            image_file = line["image"]
            qs = line["text"]
            cur_prompt = qs

            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            msg = Message()
            msg.add_message(qs)

            result = text_processor(msg.messages, mode='eval')
            input_ids = result['input_ids']
            prompt = result['prompt']
            input_ids_list.append(input_ids)
            prompt_list.append(prompt)
            idx_list.append(idx)

            image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
            image_tensor = image_processor(image)
            image_tensors_list.append(image_tensor)
            image_sizes_list.append(image.size)

        # Pad input_ids to the same length
        input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        input_ids_padded = input_ids_padded.cuda()

        image_tensors = torch.stack(image_tensors_list).half().cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids_padded,
                images=image_tensors,
                image_sizes=image_sizes_list,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for idx, prompt, output in zip(idx_list, prompt_list, outputs):
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                       "prompt": prompt,
                                       "text": output.strip(),
                                       "answer_id": ans_id,
                                       "model_id": args.model_base,
                                       "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)  # Added batch_size argument
    args = parser.parse_args()

    eval_model(args)
