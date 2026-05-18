import os
import json
import random
import datetime
import argparse
from tqdm import tqdm
import torch
from load_data import *
from utils import get_model
from ilvad import AttentionEnhancerContext

def compute_attention_heatmap(attn: torch.Tensor, scale: float) -> torch.Tensor:
    heatmap = attn
    mean, std = heatmap.mean(), heatmap.std()
    lower, upper = mean - 3 * std, mean + 3 * std
    heatmap = torch.clamp(heatmap, min=lower.item(), max=upper.item())
    heatmap = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())) * scale
    return torch.exp(heatmap)

def get_enhanced_map(output_attentions, img_start, img_end, scale, tau):

    l = len(output_attentions)
    layer_attentions = []
    all_attn = []
    img_num = img_end - img_start
    device = output_attentions[0][0].device
    num_heads = len(output_attentions[0][0][0])
    k = int(num_heads * 0.5)

    for layer in range(len(output_attentions[0])):
        attn = torch.zeros(img_num, device=device)
        for t in range(1, l):
            attentions = output_attentions[t][layer]
            img_attentions = attentions[0, :, -1, img_start:img_end]
            head_importance = img_attentions.sum(dim=-1)
            top_indices = torch.topk(head_importance, k=k).indices
            selected_attn = img_attentions[top_indices, :]
            step_saliency = torch.mean(selected_attn, dim=0)
            attn += step_saliency  
        attn = attn / (l-1)
        all_attn.append(attn)
        mean_value = attn.mean()
        binary_attn = (attn > tau * mean_value).float()
        layer_attentions.append(binary_attn)
    
    all_attn_tensor = torch.stack(all_attn)
    avg_attn = all_attn_tensor.mean(dim=0)
    avg = avg_attn.mean()
    
    bi_attn = (avg_attn > avg).float()

    layer_attentions_tensor = torch.stack(layer_attentions)
    layer_diffs = layer_attentions_tensor[1:] - layer_attentions_tensor[:-1]
    positive_diffs = torch.maximum(layer_diffs, torch.tensor(0.0, device=device)) 
    temp = torch.sum(positive_diffs, dim=0)
    enchanced_map = temp * bi_attn
    
    return compute_attention_heatmap(enchanced_map, scale)


def load_dataset(dataset_name, seed):
    if "pope" in dataset_name:
        dataset = POPE(subset=dataset_name)
    elif dataset_name == "chair":
        dataset = CHAIRBench(500)
    elif dataset_name == "llava_wild":
        dataset = LLavaWild()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return dataset

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str, help="path to the model")
    parser.add_argument("--chunk_idx", default=0, type=int, help="chunk index for parallel inference")
    parser.add_argument("--device", default="cuda:2", type=str, help="device")
    parser.add_argument("--method", default="ilvad", type=str, help="method to use for hallucination mitigation")
    parser.add_argument("--alpha", default=5, type=float, help="enhancement strength for ILVAD")
    parser.add_argument("--beta", required=True, type=float, help="control parameter of text attention for ILVAD")
    parser.add_argument("--tau", default=5, type=int, help="threshold for ILVAD")
    parser.add_argument("--T", default=10, type=int, help="first-token window for ILVAD")
    parser.add_argument("--layer_to_enhance", nargs="+", type=int, required=True,help="layers to enhance, e.g. --layer_to_enhance 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25")
    parser.add_argument("--max_new_tokens", default=512, type=int, help="max new tokens to generate")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--output_dir", default="outputs", type=str, help="output directory")
    parser.add_argument("--datasets",nargs="+",type=str,required=True,help="datasets to run, e.g. --datasets chair or --datasets pope_random pope_popular pope_adversarial")
    return parser

class HallucinationMitigation:

    def __init__(self, args):
        attn_impl = "eager"
        self.device = torch.device(args.device)
        self.processor, self.model, self.template, self.ans_start, self.num_img_tokens, self.image_token_id = get_model(args.model_path, self.device, attn_impl)
        self.model_name = os.path.basename(os.path.normpath(args.model_path))
        self.output_dir = args.output_dir
        self.method = args.method
        self.max_new_tokens = args.max_new_tokens
        self.alpha = args.alpha
        self.beta = args.beta
        self.tau = args.tau
        self.T = args.T
        self.layer_to_enhance = args.layer_to_enhance

    def infer_dataset(self, dataset_name, seed):
        output_file_path = f"{self.output_dir}/{self.model_name}/{dataset_name}_{seed}-{self.method}-{self.max_new_tokens}.json"
        complete_output_file_path = f"{self.output_dir}/{self.model_name}/{dataset_name}_{seed}-{self.method}-{self.max_new_tokens}.json"
        
        if (os.path.isfile(output_file_path) and len(open(output_file_path, "r").readlines()) > 0) \
            or (os.path.isfile(complete_output_file_path) and len(open(complete_output_file_path, "r").readlines()) > 0):
            return

        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        random.seed(seed)

        dataset = load_dataset(dataset_name, seed)    

        generate_kwargs = {
            "num_beams": 5 if "beam" in self.method else 1, 
            "do_sample": "sample" in self.method,
            "max_new_tokens": self.max_new_tokens,
        }
        
        output_file = open(output_file_path, "w")
        
        start = datetime.datetime.now()
        
        ##chair, pope and llava wild##
        for qid in tqdm(dataset.all, desc=f"Inference on {dataset_name} with seed {seed}"): 
            img, qu, gt = dataset[qid]
            prompt = self.template.format(question=qu)
            inputs = self.processor(text=prompt, images=img, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            img_start, img_end, respond_start, img_token_end = self.get_image_position(inputs['input_ids'])

            if "ilvad" in self.method:
                with torch.inference_mode():
                    outputs = self.model.generate(**inputs, max_new_tokens = self.T, output_attentions=True, output_scores=True, return_dict_in_generate=True)
                    output_attentions = outputs.attentions

                enhanced_map = get_enhanced_map(output_attentions, img_start, img_end, self.alpha, self.tau)

                enhancer = AttentionEnhancerContext(
                    model=self.model,
                    img_start=img_start,  
                    img_end=img_end,  
                    respond_start=respond_start, 
                    enhanced_map=enhanced_map,
                    layers_to_modify=self.layer_to_enhance,
                    beta = self.beta,
                    model_name=self.model_name
                )
                with enhancer:
                    with torch.inference_mode():
                        outputs = self.model.generate(**inputs, **generate_kwargs, output_scores=True, return_dict_in_generate=True)
            else:
                with torch.inference_mode():
                    outputs = self.model.generate(**inputs, **generate_kwargs, output_attentions=True, output_scores=True, return_dict_in_generate=True)

            
            gen_answer = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
            gen_answer = gen_answer.split(self.ans_start)[1].strip() if self.ans_start is not None else gen_answer.strip()
            output_file.write(json.dumps({"question_id": qid, "text": gen_answer, "gt": gt, "question": qu})+"\n")
            output_file.flush()
            os.fsync(output_file.fileno())

        end = datetime.datetime.now()
        print(f"Method: {self.method}\tInference Time: {(end-start).seconds} s")
    
    def get_image_position(self, input_ids):
        if self.num_img_tokens is None:
            self.num_img_tokens = (input_ids==self.image_token_id).nonzero().shape[0]

        if self.image_token_id is None:
            indices = torch.where(input_ids[0] == self.image_token_id )
            img_start = indices[0][0].item()
            img_end = indices[0][-1].item()
            respond_start = img_end + 2
            img_token_end = (input_ids[0] == self.image_token_id).nonzero()[-1, 1].item() + 1
        else:
            img_start = (input_ids==self.image_token_id).nonzero()[0,1].item()
            respond_start = input_ids.shape[1] - (input_ids==self.image_token_id).nonzero().shape[0] + self.num_img_tokens
            img_token_end = (input_ids==self.image_token_id).nonzero()[-1, 1].item() + 1
            img_end = img_start + self.num_img_tokens

        return img_start, img_end, respond_start, img_token_end


if __name__ == "__main__":
    args = argument_parser().parse_args()

    hal_test = HallucinationMitigation(args)

    for dataset_name in args.datasets:
        hal_test.infer_dataset(dataset_name, args.seed)