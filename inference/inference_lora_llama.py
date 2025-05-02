import sys
sys.path.append('./lora')
from finetune_lora_llama_abstract import extract_abstract, drop_indices,load_dataset
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import pandas as pd
import json,gc,os,torch
from dataclasses import dataclass, asdict

@dataclass
class Config:
    output_dir: str = "output"
    checkpoint: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Update to LLaMA 3 checkpoint
    experiment_index: str = '3'
    plos_lora_checkpoint: str = "linf545/LLaMA_lora_lr1e5_epo1_rank8_PLOS_0404"
    elife_lora_checkpoint: str = "linf545/LLaMA_lora_lr1e5_epo2_rank8_eLife_0425"
    dataset_name: str = "BioLaySumm/BioLaySumm2025-PLOS"
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


def summary_length(dataset_name):
    if dataset_name == "BioLaySumm/BioLaySumm2025-PLOS":
        return '100-300 words'
    if dataset_name == "BioLaySumm/BioLaySumm2025-eLife":
        return '200-600 words'


def format_inference_prompt(sample):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are an expert science communicator. Your task is to generate a **clear, accurate, and formal** summary of biomedical research articles.
    The summary should be **accessible to a general audience** while maintaining scientific rigor.<|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    Title: {sample['title']}
    Abstract: {sample['abstract']}

    Provide a **formal summary** of the article in {summary_word_len}. **Do not include explanations, self-reflections, or additional notes.** 
    Keep the response strictly to the summary.The output should begin directly with the summary text itself.<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
    return {
        "input_text": prompt,  # Model input (including expected output)
    }

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
    autoconfig = AutoConfig.from_pretrained(config.checkpoint)
    autoconfig.rope_scaling = {"type": "linear", "factor": 2.0}  
    config.save("./configfile/inference_experiment_%s_config.json"%(config.experiment_index))
    base_model = AutoModelForCausalLM.from_pretrained(config.checkpoint, config=autoconfig, torch_dtype="auto", device_map="cuda")

    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(config.dataset_name)
    dataset = dataset.map(extract_abstract)
    dataset.column_names

    for j in ["BioLaySumm/BioLaySumm2025-PLOS", "BioLaySumm/BioLaySumm2025-eLife"]:
        if j== "BioLaySumm/BioLaySumm2025-PLOS":
            lora_weights= config.plos_lora_checkpoint
        else: 
            lora_weights= config.elife_lora_checkpoint
    
        model = PeftModel.from_pretrained(base_model, lora_weights,
                                            device_map="auto",
                                            torch_dtype=torch.bfloat16) # bf16 if using H100 )
        dataset = load_dataset(j)
        dataset = dataset.map(extract_abstract)
        dataset.column_names

        if j == "BioLaySumm/BioLaySumm2025-PLOS":
            plos_drop_dict={'train':[[725, 1939, 4226, 4842, 5991, 6310, 12050, 13498, 14104, 14199, 18921, 21808, 22922]],'validation':[],'test':[]} # drop due to inccorrect abstract
            dataset = drop_indices(dataset, plos_drop_dict)

        for i in ['test', 'validation']:
            selected_set = dataset[i]
            summary_word_len = summary_length(j)
            val_set=selected_set
            formatted_val = val_set.map(format_inference_prompt, remove_columns=dataset[i].column_names)
            print('start to generate output')
            generated_val = formatted_val.map(generate_output)
            print('writing outputs')
            output_path = "./output/generated_summaries/indexed_experiments/experiment%s/%s_%s_summaries.txt" % (config.experiment_index,j.split('-')[1], i)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                for line in generated_val['summary']:
                    f.write(line + "\n")
            generated_val.to_csv("./output/generated_summaries/indexed_experiments/experiment%s/%s_%s_check.csv"%(config.experiment_index,j.split('-')[1],i))
            del  formatted_val, generated_val, selected_set
            free_cuda()
        del model,dataset
        free_cuda()
    del tokenizer, base_model
    free_cuda()