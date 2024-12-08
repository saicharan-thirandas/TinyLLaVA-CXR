# Updated collate_fn
import argparse
import time

import torch
import os
import json
from tqdm import tqdm
import shortuuid

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *

from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, text_processor, image_processor):
        self.questions = questions
        self.image_folder = image_folder
        self.text_processor = text_processor
        self.image_processor = image_processor

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = self.image_processor(image)
        
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        msg = Message()
        msg.add_message(qs)
        #print(prompt)
        result = self.text_processor(msg.messages, mode='eval')
        input_ids = result['input_ids']

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)  # Pad input_ids for batch
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# Updated create_data_loader
def create_data_loader(questions, image_folder, text_processor, image_processor, batch_size=1, num_workers=16):
    dataset = CustomDataset(questions, image_folder, text_processor, image_processor)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn
    )
    return data_loader


# Updated eval_model
def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)
    
    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    image_processor = ImagePreprocess(image_processor, data_args)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    # Update DataLoader to accept batch_size from args
    data_loader = create_data_loader(
        questions, args.image_folder, text_processor, image_processor, batch_size=args.batch_size,  num_workers = args.num_workers
    )

    model.to(device='cuda')
    for batch, lines in tqdm(zip(data_loader, split_list(questions, len(data_loader))), total=len(data_loader)):
        input_ids, image_tensors, image_sizes = batch
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        image_tensors = image_tensors.to(dtype=torch.float16, device='cuda', non_blocking=True)
        outputs = []

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensors,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                image_sizes=image_sizes,
                use_cache=True
            )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for line, output in zip(lines, outputs):
            idx = line["question_id"]
            cur_prompt = line["text"]
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({
                "question_id": idx,
                "prompt": cur_prompt,
                "text": output.strip(),
                "answer_id": ans_id,
                "model_id": args.model_base,
                "metadata": {}
            }) + "\n")
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llama")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--image_aspect_ratio", type=str, default="pad")
    parser.add_argument("--batch-size", type=int, default=1)  # Add batch size argument
    parser.add_argument("--num-workers", type=int, default=16)  # Add batch size argument
    args = parser.parse_args()

    eval_model(args)
