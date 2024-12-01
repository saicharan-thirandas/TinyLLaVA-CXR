'''
@Description: 
@Author: jiajunlong
@Date: 2024-06-19 19:30:17
@LastEditTime: 2024-06-19 19:32:47
@LastEditors: jiajunlong
'''
import argparse
import hashlib
import json
from pathlib import Path
import time
from threading import Thread
import logging

import gradio as gr
import torch
from transformers import TextIteratorStreamer

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *

DEFAULT_MODEL_PATH = "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B"


block_css = """

#buttons button {
    min-width: min(120px,100%);
}
"""
title_markdown = """
# TinyLLaVA-CXR: Advancing Portable Chest X-Ray Diagnostics with Efficient LLM-Driven Image Reasoning
[[Code](https://github.com/DLCV-BUAA/TinyLLaVABench)] | 📚 [[Paper](https://arxiv.org/pdf/2402.14289.pdf)]
"""
tos_markdown = """
### Terms of use
By using this service, users are required to agree to the following terms:
The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes.
For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.
"""
learn_more_markdown = """
### License
The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.
"""
ack_markdown = """
### Acknowledgement
The template for this web demo is from [LLaVA](https://github.com/haotian-liu/LLaVA), and we are very grateful to LLaVA for their open source contributions to the community!
"""


def regenerate(state, image_process_mode):
    state.messages[-1]['value'] = None
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None)


def clear_history():
    state = Message()
    return (state, state.to_gradio_chatbot(), "", None)


def add_text(state, text, image, image_process_mode):
    if len(text) <= 0 and image is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None)

    text = text[:1536]  # Hard cut-off
    if image is not None:
        text = text[:1200]  # Hard cut-off for images
        if "<image>" not in text:
            # text = '<Image><image></Image>' + text
            text = text + "\n<image>"
        if len(state.images) > 0:
            state = Message()
        state.add_image(image, len(state.messages))
    state.add_message(text, None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "", None)


def load_demo():
    state = Message()
    return state


@torch.inference_mode()
def get_response(params):
    input_ids = params["input_ids"]
    prompt = params["prompt"]
    images = params.get("images", None)
    num_image_tokens = 0
    if images is not None and len(images) > 0:
        if len(images) > 0:
            # image = [load_image_from_base64(img) for img in images][0]
            image = images[0][0]
            image = image_processor(image)
            image = image.unsqueeze(0).to(model.device, dtype=torch.float16)
            num_image_tokens = getattr(model.vision_tower._vision_tower, "num_patches", 336)
        else:
            image = None
        image_args = {"images": image}
    else:
        image = None
        image_args = {}

    temperature = float(params.get("temperature", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_context_length = getattr(model.config, "max_position_embeddings", 2048)
    max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
    stop_str = params.get("stop", None)
    do_sample = True if temperature > 0.001 else False
    logger.info(prompt)
    input_ids = input_ids.unsqueeze(0).to(model.device)
    # keywords = [stop_str]

    # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15
    )

    max_new_tokens = min(
        max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens
    )

    if max_new_tokens < 1:
        yield json.dumps(
            {
                "text": prompt
                + "Exceeds max token length. Please start a new conversation, thanks.",
                "error_code": 0,
            }
        ).encode() + b"\0"
        return

    generate_kwargs = dict(
        inputs=input_ids,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        use_cache=True,
        pad_token_id = tokenizer.eos_token_id,
        **image_args,
    )
    thread = Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()
    logger.debug(prompt)
    logger.debug(generate_kwargs)
    generated_text = prompt
    for new_text in streamer:
        generated_text += new_text
        # print(f"new_text:{new_text}")
        if generated_text.endswith(stop_str):
            generated_text = generated_text[: -len(stop_str)]
        yield json.dumps({"text": generated_text, "error_code": 0}).encode()


def http_bot(state, temperature, top_p, max_new_tokens):
    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot())
        return
    
    
    images = state.images
    result = text_processor(state.messages, mode='eval')
    prompt = result['prompt']
    input_ids = result['input_ids']
    pload = {
        "model": model_name,
        "prompt": prompt,
        "input_ids": input_ids,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": min(int(max_new_tokens), 1536),
        "stop": (
            text_processor.template.separator.apply()[1]
        ), "images": images}

    state.messages[-1]['value'] = "▌"
    yield (state, state.to_gradio_chatbot())

    # for stream
    output = get_response(pload)
    for chunk in output:
        if chunk:
            data = json.loads(chunk.decode())
            if data["error_code"] == 0:
                output = data["text"][len(prompt) :].strip()
                state.messages[-1]['value'] = output + "▌"
                yield (state, state.to_gradio_chatbot())
            else:
                output = data["text"] + f" (error_code: {data['error_code']})"
                state.messages[-1]['value'] = output
                yield (state, state.to_gradio_chatbot())
                return
            time.sleep(0.03)

    state.messages[-1]['value'] = state.messages[-1]['value'][:-1]
    yield (state, state.to_gradio_chatbot())


def build_demo():
    textbox = gr.Textbox(
        show_label=False, placeholder="Enter text and press ENTER", container=False
    )
    with gr.Blocks(title="TinyLLaVA", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()
        gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=5):
                with gr.Row(elem_id="Model ID"):
                    gr.Dropdown(
                        choices=[DEFAULT_MODEL_PATH.split('/')[-1]],
                        value=DEFAULT_MODEL_PATH.split('/')[-1],
                        interactive=True,
                        label="Model ID",
                        container=False,
                    )
                imagebox = gr.Image(type="pil")
                image_process_mode = gr.Radio(
                    ["Crop", "Resize", "Pad", "Default"],
                    value="Default",
                    label="Preprocess for non-square image",
                    visible=False,
                )

                # cur_dir = os.path.dirname(os.path.abspath(__file__))
                cur_dir = Path(__file__).parent
                gr.Examples(
                    examples=[
                        [
                            f"{cur_dir}/examples/extreme_ironing.jpg",
                            "What is unusual about this image?",
                        ],
                        [
                            f"{cur_dir}/examples/waterview.jpg",
                            "What are the things I should be cautious about when I visit here?",
                        ],
                    ],
                    inputs=[imagebox, textbox],
                )

                with gr.Accordion("Parameters", open=False) as _:
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.2,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )
                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        interactive=True,
                        label="Top P",
                    )
                    max_output_tokens = gr.Slider(
                        minimum=0,
                        maximum=1024,
                        value=512,
                        step=64,
                        interactive=True,
                        label="Max output tokens",
                    )

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(elem_id="chatbot", label="Chatbot", height=550)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary")
                with gr.Row(elem_id="buttons") as _:
                    regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=True)
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=True)

        gr.Markdown(tos_markdown)
        gr.Markdown(learn_more_markdown)
        gr.Markdown(ack_markdown)

        regenerate_btn.click(
            regenerate,
            [state, image_process_mode],
            [state, chatbot, textbox, imagebox],
            queue=False,
        ).then(
            http_bot, [state, temperature, top_p, max_output_tokens], [state, chatbot]
        )

        clear_btn.click(
            clear_history, None, [state, chatbot, textbox, imagebox], queue=False
        )

        textbox.submit(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox],
            queue=False,
        ).then(
            http_bot, [state, temperature, top_p, max_output_tokens], [state, chatbot]
        )

        submit_btn.click(
            add_text,
            [state, textbox, imagebox, image_process_mode],
            [state, chatbot, textbox, imagebox],
            queue=False,
        ).then(
            http_bot, [state, temperature, top_p, max_output_tokens], [state, chatbot]
        )

        demo.load(load_demo, None, [state], queue=False)
    return demo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--share", default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default="phi")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_PATH.split('/')[-1])
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info(gr.__version__)
    args = parse_args()
    model_name = args.model_name
    model, tokenizer, image_processor, context_len = load_pretrained_model(
        args.model_path,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit
    )
    model.to(args.device)
    image_processor = ImagePreprocess(image_processor, model.config)
    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    demo = build_demo()
    demo.queue()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
