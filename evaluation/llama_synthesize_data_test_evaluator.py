import sys
sys.path.append('./lora')
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


def prompt_generate_normal_examples(sample):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are an expert science communicator. Your task is to generate a **clear, accurate, and formal** summary of biomedical research articles.
    The summary should be **accessible to a general audience** while maintaining scientific rigor.

    <|start_header_id|>user<|end_header_id|>
    Title: {sample['title']}
    Abstract: {sample['abstract']}

    Provide a **formal summary** of the article in 100-300 words. **Do not include explanations, self-reflections, or additional notes.** Keep the response strictly to the summary.
    <|start_header_id|>assistant<|end_header_id|>
    """
    return {
        "input_text": prompt,  # Model input (including expected output)
    }


def prompt_generate_bad_relavance(sample):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are a science communicator. Your task is to provide a summary of biomedical research articles by focusing on general themes and broad scientific context.
    There is **no need to focus closely on the study details**; feel free to summarize based on general knowledge in similar research areas.

    <|start_header_id|>user<|end_header_id|>
    Title: {sample['title']}
    Abstract: {sample['abstract']}

    Provide a **formal summary** of the article in 100-300 words. You may focus on the overall topic area rather than specific results or methods. Avoid overloading with precise details.
    <|start_header_id|>assistant<|end_header_id|>
    """
    return {
        "input_text": prompt,  # Model input (including expected output)
    }

def prompt_generate_bad_redability(sample):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are a science writer. Your task is to provide a detailed summary of biomedical articles, including as much information as possible.
    Do not worry about simplifying the language or organizing the text for easy understanding.

    <|start_header_id|>user<|end_header_id|>
    Title: {sample['title']}
    Abstract: {sample['abstract']}

    Provide a **complete and information-rich summary** of the article in 100-300 words. You may include technical phrasing, long sentences, and complex descriptions. Prioritize detail over clarity.
    <|start_header_id|>assistant<|end_header_id|>
    """
    return {
        "input_text": prompt,  # Model input (including expected output)
    }


def prompt_generate_bad_factuality(sample):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are a science communicator. Your task is to write summaries of biomedical research articles by interpreting the likely implications and goals, even if not explicitly stated.
    You may extrapolate or hypothesize based on the abstract.

    <|start_header_id|>user<|end_header_id|>
    Title: {sample['title']}
    Abstract: {sample['abstract']}

    Provide a **summary** of the article in 100-300 words. You may include **reasonable assumptions, interpretations, or inferred conclusions** to enhance the summary, even if not directly stated in the abstract.
    <|start_header_id|>assistant<|end_header_id|>
    """
    return {
        "input_text": prompt,  # Model input (including expected output)
    }


def prompt_generate_bad(sample):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are a deliberately ineffective science communicator. Your task is to generate an example of a poorly written summary of biomedical research. This summary should reflect common mistakes in science communication, such as vague language, poor structure, and misuse of terminology. The summary may also include minor factual inaccuracies or exaggerated claims to illustrate how misleading summaries might appear. This output will be used strictly for educational comparison with well-written summaries.

    <|start_header_id|>user<|end_header_id|>
    Title: {sample['title']}
    Abstract: {sample['abstract']}

    Provide a **poor-quality summary** of the article in 100-300 words, reflecting issues like lack of clarity, overgeneralization, or scientific inaccuracy (intended for contrastive purposes only).At least some summary need to generated.**Do not include explanations, self-reflections, or additional notes.** Keep the response strictly to the summary.
    <|start_header_id|>assistant<|end_header_id|>
    """
    return {
        "input_text": prompt,  # Model input (including expected output)
    }

def generate_output(sample):
    inputs = tokenizer(sample["input_text"], return_tensors="pt",  truncation=True,max_length=1024)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device) 
    output_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=3000,do_sample=False, pad_token_id=tokenizer.eos_token_id)
    sample["summary"] = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    sample["summary"] = sample["summary"].replace("poorly written", "")
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
val_subset = val_set.select(range(20))
formatted_val = val_subset.map(prompt_generate_normal_examples, remove_columns=dataset["validation"].column_names)
generated_val = formatted_val.map(generate_output)
generated_val.to_parquet("./output/synthesized_data/normal_summaries.parquet")

'''bad_relavance_set = val_set.select(range(20))
formatted_bad_relavance = bad_relavance_set.map(prompt_generate_bad_relavance, remove_columns=dataset["validation"].column_names)
generated_bad_relavance = formatted_bad_relavance.map(generate_output)
generated_bad_relavance.to_parquet("./output/synthesized_data/bad_relavance_summaries.parquet")

bad_readability_set = val_set.select(range(20))
formatted_bad_readability = bad_readability_set.map(prompt_generate_bad_redability, remove_columns=dataset["validation"].column_names)
generated_bad_readability = formatted_bad_readability.map(generate_output)
generated_bad_readability.to_parquet("./output/synthesized_data/bad_readability_summaries.parquet")


bad_factuality_set = val_set.select(range(20))
formatted_bad_factuality = bad_factuality_set.map(prompt_generate_bad_factuality, remove_columns=dataset["validation"].column_names)
generated_bad_factuality = formatted_bad_factuality.map(generate_output)
generated_bad_factuality.to_parquet("./output/synthesized_data/bad_factuality_summaries.parquet")
'''
bad_set = val_set.select(range(20))
formatted_bad = bad_set.map(prompt_generate_bad, remove_columns=dataset["validation"].column_names)
generated_bad = formatted_bad.map(generate_output)
generated_bad.to_parquet("./output/synthesized_data/bad_summaries.parquet")