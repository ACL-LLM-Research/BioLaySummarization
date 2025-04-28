import sys
sys.path.append('./lora')
from finetune_lora_llama_abstract import extract_abstract, drop_indices,load_dataset
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import pandas as pd
import json
from dataclasses import dataclass, asdict
import os, gc 
import torch

@dataclass
class Config:
    output_dir: str = "output"
    checkpoint: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Update to LLaMA 3 checkpoint
    experiment_index: str = '8'
    #experiment_name: str = "LLaMA_base_PLOS_0329"
    #dataset_name: str = "BioLaySumm/BioLaySumm2025-PLOS"
    max_new_tokens: int= 800
    num_beams: int= 4
    input_max_length: int = 2048
    def save(self, path: str):
          with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)
    @staticmethod
    def load(path: str):
        with open(path, "r") as f:
            data = json.load(f)
        return Config(**data)
    

def format_inference_prompt(sample):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are an expert science communicator. Your task is to generate a **clear, accurate, and formal** summary of biomedical research articles.
    The summary should be **accessible to a general audience** while maintaining scientific rigor.<|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    Title: {sample['title']}
    Article: {sample['article']}

    Provide a **formal summary** of the article in {summary_word_len}. **Do not include explanations, self-reflections, or additional notes.** 
    Keep the response strictly to the summary.The output should begin directly with the summary text itself.<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
    return {
        "input_text": prompt,  # Model input (including expected output)
    }

def summary_length(dataset_name):
    if dataset_name == "BioLaySumm/BioLaySumm2025-PLOS":
        return '100-300 words'
    if dataset_name == "BioLaySumm/BioLaySumm2025-eLife":
        return '200-600 words'


def generate_output(sample):
    inputs = tokenizer(sample["input_text"], return_tensors="pt",  truncation=True,max_length=config.input_max_length)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device) 
    output_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=config.max_new_tokens,num_beams=config.num_beams,
                                do_sample=False, temperature=None,top_p=None,pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if "assistant\n" in decoded:
        summary = decoded.split("assistant\n", 1)[-1].strip()
    else:
        summary = decoded.strip()
    sample["summary"] = summary
    return sample

def free_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


if __name__ == "__main__":
    config = Config()
    config.save("./configfile/inference_experiment_%s_config.json"%(config.experiment_index))
    #model = PeftModel.from_pretrained(config.checkpoint, "linf545/%s"%(config.experiment_name))
    #model = model.to("cuda")
    #tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)

    autoconfig = AutoConfig.from_pretrained(config.checkpoint)
    autoconfig .rope_scaling = {"type": "linear", "factor": 2.0}  
    model = AutoModelForCausalLM.from_pretrained(config.checkpoint, config=autoconfig, 
                                                device_map="auto",
                                                torch_dtype=torch.bfloat16) # bf16 if using H100 
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    for j in ["BioLaySumm/BioLaySumm2025-PLOS", "BioLaySumm/BioLaySumm2025-eLife"]:
        dataset = load_dataset(j)
        #dataset = dataset.map(extract_abstract)
        dataset.column_names

        if j == "BioLaySumm/BioLaySumm2025-PLOS":
            plos_drop_dict={'train':[[725, 1939, 4226, 4842, 5991, 6310, 12050, 13498, 14104, 14199, 18921, 21808, 22922]],'validation':[],'test':[]} # drop due to inccorrect abstract
            dataset = drop_indices(dataset, plos_drop_dict)

        for i in ['test', 'validation']:
            val_set=dataset[i]

            summary_word_len = summary_length(j)
            formatted_val = val_set.map(format_inference_prompt, remove_columns=dataset[i].column_names)
            #formatted_test = test_set.map(format_prompt, remove_columns=dataset["test"].column_names)

            #test_case = formatted_val.select(range(5))
            #result=test_case.map(generate_output)
            #generated_val = result
            print('start to generate output')
            generated_val = formatted_val.map(generate_output)
            print('writing outputs')
            output_path = "./output/generated_summaries/indexed_experiments/experiment%s/%s_%s_summaries.txt" % (config.experiment_index,j.split('-')[1], i)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for line in generated_val['summary']:
                    f.write(line + "\n")
            
            generated_val.to_parquet("./output/generated_summaries/indexed_experiments/experiment%s/%s_%s_check.csv"%(config.experiment_index,j.split('-')[1],i))

            del formatted_val, generated_val, val_set
            free_cuda()

    del model, tokenizer, 
    free_cuda()
