import os
from dataclasses import dataclass
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
#from torch.amp import GradScaler

#scaler = GradScaler("cuda")

class Config:
    output_dir: str = "output"
    checkpoint: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Update to LLaMA 3 checkpoint
    max_length: int = 2048
    optim_type: str = "adamw_torch"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4  
    per_device_eval_batch_size: int = 2
    n_epochs: int = 3
    freeze_layers: int = 20  # other option 16,20,24
    lr: float = 2e-4
    #warmup_steps: int = 20
    lora_r: int = 32
    lora_alpha: float = lora_r * 2
    lora_dropout: float = 0.1
    lora_bias: str = "none"


config = Config()

training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    report_to="none",
    num_train_epochs=config.n_epochs,
    per_device_train_batch_size=config.per_device_train_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    logging_steps=10,
    eval_strategy="epoch",
    #eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=100,
    optim=config.optim_type,
    fp16=True,
    #bf16=True,
    learning_rate=config.lr,
    #warmup_steps=config.warmup_steps,
    #ddp_find_unused_parameters=False,
    #gradient_checkpointing=True,
    logging_dir="./logs",
    label_names=["labels"]
)

lora_config = LoraConfig(
    r=config.lora_r,  # Higher rank for better adaptation
    lora_alpha=config.lora_alpha,  # Scaling factor, higher for better adaptation
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Apply LoRA to all attention layers
    layers_to_transform=[i for i in range(32) if i >= config.freeze_layers],  # Apply LoRA to higher layers (assuming 80-layer model)
    lora_dropout=config.lora_dropout,  # Higher dropout to prevent overfitting
    bias=config.lora_bias,  # No additional biases
    task_type=TaskType.CAUSAL_LM,  # Task type for text generation
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    config.checkpoint,
    #device_map="auto",
).to(device)
model.config.rope_scaling = {"type": "linear", "factor": 2.0}
model.config.use_cache = False
model = get_peft_model(model, lora_config)




##load data
dataset = load_dataset("BioLaySumm/BioLaySumm2025-PLOS")
#dataset['train'][0]['article']
#dataset['train'][0]['summary']


def format_prompt(sample):
    """
    Converts dataset sample into an instruction-tuned format for LLaMA3.
    """
    prompt = f"""
    [SYSTEM]  
    You are a skilled science communicator. Your task is to generate a clear, concise summary of biomedical research articles for a general audience.

    [USER]  
    Generate a summary of the following biomedical article, ensuring accessibility for non-experts.

    Title: {sample['title']}  
    Full Text: {sample['article']}  

    [ASSISTANT]  
    {sample['summary']}
    """
    return {
        "input_text": prompt,  # Model input (including expected output)
    }

# Apply the function to the dataset

#test_train = dataset["train"].select(range(50))  # First n samples
#test_val = dataset["validation"].select(range(50))  # First n validation samples
test_train =dataset["train"]
test_val=dataset["validation"]

formatted_train = test_train.map(format_prompt, remove_columns=dataset["train"].column_names)
formatted_val = test_val.map(format_prompt, remove_columns=dataset["validation"].column_names)

tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" 

def tokenize_data(sample):
    encodings = tokenizer(
        sample["input_text"], 
        truncation=True, 
        padding="max_length", 
        max_length=config.max_length
    )
    input_ids = torch.tensor(encodings["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(encodings["attention_mask"], dtype=torch.long)
    # Ensure labels are the same as input_ids but mask padding tokens
    labels = input_ids.clone()
    labels[:-1] = input_ids[1:]  # Shift all tokens left
    labels[-1] = tokenizer.pad_token_id  # Ensure last token is padding
    labels[labels == tokenizer.pad_token_id] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels  # Should now always be 1D
    }

tokenized_train = formatted_train.map(tokenize_data, batched=True, remove_columns=["input_text"])
tokenized_val = formatted_val.map(tokenize_data, batched=True, remove_columns=["input_text"])


# Ensure the format is correct for training
#tokenized_train.set_format("torch", device="cuda", columns=["input_ids", "attention_mask"])
#tokenized_val.set_format("torch", device="cuda", columns=["input_ids", "attention_mask"])

tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


'''
class DataCollatorForSeq2SeqWithDevice(DataCollatorForSeq2Seq):
    def __call__(self, features):
        batch = super().__call__(features)
        return {k: v.to(device) for k, v in batch.items()}  # Move batch to GPU

data_collator = DataCollatorForSeq2SeqWithDevice(tokenizer, padding=True)
'''
data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True) # even out batch length


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
)

#torch.cuda.reset_peak_memory_stats()

trainer.train()

#peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert bytes to GB
#print(f"Peak GPU memory usage: {peak_memory:.2f} GB")



