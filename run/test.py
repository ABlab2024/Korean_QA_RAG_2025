import argparse
import json, re
import tqdm

import torch
import numpy
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.data2 import CustomDataset

# torchDynamo(최적화 기능) 캐시 늘리기
torch._dynamo.config.cache_size_limit = 128


# fmt: off
parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--input", type=str, required=True, help="input filename")
g.add_argument("--output", type=str, required=True, help="output filename")
g.add_argument("--model_id", type=str, required=True, help="huggingface model id")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer")
g.add_argument("--device", type=str, required=True, help="device to load the model")
g.add_argument("--token", type=str, help="Hugging Face token for accessing gated models")
g.add_argument("--save_model_path", type=str, help="local path to save the model")
# fmt: on


def main(args):
    # Prepare model loading kwargs
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": args.device,
    }
    if args.token:
        model_kwargs["token"] = args.token
        model_kwargs["cache_dir"] = args.save_model_path

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        **model_kwargs
    )
    model.eval()

    if args.tokenizer == None:
        args.tokenizer = args.model_id
    
    # Prepare tokenizer loading kwargs
    tokenizer_kwargs = {}
    if args.token:
        tokenizer_kwargs["token"] = args.token
        if "LG" in args.model_id:
            tokenizer_kwargs["enable_thinking"] = False
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        **tokenizer_kwargs
    )
    # 모델의 원래 pad_token을 사용하거나, 없으면 eos_token 사용
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>") if tokenizer.convert_tokens_to_ids("<|eot_id|>") else tokenizer.convert_tokens_to_ids("<|endoftext|>")
    ]

    file_test = args.input
    dataset = CustomDataset(file_test, tokenizer)

    with open(file_test, "r") as f:
        result = json.load(f)

    for idx in tqdm.tqdm(range(len(dataset))):
        inp = dataset[idx]
        outputs = model.generate(
            inp.to(args.device).unsqueeze(0),
            max_new_tokens=1024,
            eos_token_id=terminators,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.05,
            temperature=0.6,        # 0.7       # do_sample=True일때만 적용
            top_p=0.95,             # 0.8
            do_sample=True,
            # num_beams=5,             # beam search로 5개의 후보를 동시에 탐색해서 최적 문장 선택(메모리 소비 커짐)
        )
        # print(tokenizer.decode(inp, skip_special_tokens=True))
        output_text = tokenizer.decode(outputs[0][inp.shape[-1]:], skip_special_tokens=True)
        
        # 출력에서 "답변: " 접두어 제거
        if output_text.startswith("답변: "):
            output_text = output_text[4:]
        elif output_text.startswith("답변:"):
            output_text = output_text[3:]
        
        # 출력에서 불필요한 단어 제거
        output_text = re.sub(r"\{?답변\}? ?\:?", '', output_text)
        output_text = re.sub(r"\{?해결\}? ?\:?", '', output_text)
        output_text = re.sub(r"\{?이유\}? ?\:?", '', output_text)
        output_text = re.sub(r"\{?질문\}? ?\:?", '', output_text)
        
        # print(output_text)
        result[idx]["output"] = {"answer": output_text}

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    exit(main(parser.parse_args()))
