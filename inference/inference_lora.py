import sys
sys.path.append('../lora')

from finetune_lora_llama_abstract import extract_abstract, drop_indices,config,load_dataset
from peft import PeftModel
from transformers import AutoTokenizer
import pandas as pd


def format_inference_prompt(sample):
    prompt = f"""
    [SYSTEM]  
    You are a skilled science communicator. Your task is to generate a plain-language summary of biomedical research articles, making them accessible to a general audience without specialized knowledge.
    [USER]  
    Generate a plain-language summary based on the following title and abstract, ensuring clarity and accessibility to a non-expert audience.
    Title: {sample['title']}  
    Abstract: {sample['abstract']}  

    [ASSISTANT]  
    """
    return {
        "input_text": prompt,  # Model input (including expected output)
    }

def generate_output(sample):
    inputs = tokenizer(sample["input_text"], return_tensors="pt", padding="longest", truncation=True,max_length=1024)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device) 
    output_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
    sample["summary"] = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return sample

model = PeftModel.from_pretrained(config.checkpoint, "linf545/%s"%(config.experiment_name))
model = model.to("cuda")
tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)

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


generated_val = formatted_val.map(generate_output)


df = pd.DataFrame(generated_val)
df.to_csv("./output/generated_summaries.csv", index=False)

