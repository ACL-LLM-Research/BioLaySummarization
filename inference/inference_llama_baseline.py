import sys
sys.path.append('../lora')
from finetune_lora_llama_abstract import extract_abstract, drop_indices,load_dataset
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import pandas as pd



class Config:
    output_dir: str = "output"
    checkpoint: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Update to LLaMA 3 checkpoint
    experiment_name: str = "LLaMA_lora_PLOS_0312"
    dataset_name: str = "BioLaySumm/BioLaySumm2025-PLOS"
    max_length: int = 2048
    optim_type: str = "adamw_torch"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4  
    per_device_eval_batch_size: int = 2
    n_epochs: int = 3
    freeze_layers: int = 20  # other option 16,20,24
    lr: float = 2e-4
    #warmup_steps: int = 20
    lora_r: int = 16
    lora_alpha: float = lora_r * 2
    lora_dropout: float = 0.1
    lora_bias: str = "none"


def format_inference_prompt(sample):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are an expert science communicator. Your task is to generate a **clear, accurate, and formal** summary of biomedical research articles.
    The summary should be **accessible to a general audience** while maintaining scientific rigor.

    <|start_header_id|>user<|end_header_id|>
    Title: {sample['title']}
    Abstract: {sample['abstract']}

    Provide a **formal summary** of the article in 200-400 words. **Do not include explanations, self-reflections, or additional notes.** Keep the response strictly to the summary.
    <|start_header_id|>assistant<|end_header_id|>
    """
    return {
        "input_text": prompt,  # Model input (including expected output)
    }

def generate_output(sample):
    inputs = tokenizer(sample["input_text"], return_tensors="pt",  truncation=True,max_length=1024)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device) 
    output_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=400,do_sample=False, pad_token_id=tokenizer.eos_token_id)
    sample["summary"] = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return sample


config = Config()
#model = PeftModel.from_pretrained(config.checkpoint, "linf545/%s"%(config.experiment_name))
#model = model.to("cuda")
#tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)

autoconfig = AutoConfig.from_pretrained(config.checkpoint)
autoconfig .rope_scaling = {"type": "linear", "factor": 2.0}  
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", config=autoconfig, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset(config.dataset_name)
dataset = dataset.map(extract_abstract)
dataset.column_names

if config.dataset_name == "BioLaySumm/BioLaySumm2025-PLOS":
    plos_drop_dict={'train':[[725, 1939, 4226, 4842, 5991, 6310, 12050, 13498, 14104, 14199, 18921, 21808, 22922]],'validation':[],'test':[]} # drop due to inccorrect abstract
    dataset = drop_indices(dataset, plos_drop_dict)

val_set=dataset["validation"]
#test_set=dataset["test"]


formatted_val = val_set.map(format_inference_prompt, remove_columns=dataset["validation"].column_names)
#formatted_test = test_set.map(format_prompt, remove_columns=dataset["test"].column_names)


#tokenized_val = formatted_val.map(tokenizer, batched=True, remove_columns=["input_text"])
#tokenized_test = formatted_test.map(tokenizer, batched=True, remove_columns=["input_text"])


#test_case = formatted_val.select(range(2))
#result=test_case.map(generate_output)

generated_val = formatted_val.map(generate_output)


generated_val.to_parquet("./output/test_summaries.parquet")
#test= load_dataset("parquet", data_files="./output/test_summaries.parquet")