import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration

def get_model(model_path, device, attn_implementation="eager"):
    model_kwargs = {
        "torch_dtype": torch.float16,
        "attn_implementation": attn_implementation,
        "low_cpu_mem_usage": True,
    }
    if "llava-v1.6" in model_path or "llava-next" in model_path:
        processor = LlavaNextProcessor.from_pretrained(model_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        model = torch.compile(model)  # 需 PyTorch 2.0+
        conversation = [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
                }]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "{question}"},
                    {"type": "image"},
                ],
            },
        ]
        template = processor.apply_chat_template(conversation, add_generation_prompt=True)
        ans_start = "ASSISTANT:" if "llava-v1.6" in model_path else "assistant"
        num_img_tokens = None   # TODO: get the num_img_tokens and image_start for the template
        image_token_id = model.config.image_token_index
    else:
        processor = AutoProcessor.from_pretrained(model_path)
        model = LlavaForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        model = torch.compile(model)  # 需 PyTorch 2.0+
        conversation = [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
                }]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "{question}"},
                ],
            },
        ]
        template = processor.apply_chat_template(conversation, add_generation_prompt=True)
        ans_start = "ASSISTANT:"
        num_img_tokens = 576  
        image_token_id = model.config.image_token_index
    model.to(device)
    return processor, model, template, ans_start, num_img_tokens, image_token_id

def get_layers(model):
    
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    if hasattr(model, "language_model"):
        return model.language_model.model.layers
    else:
        return model.model.layers
