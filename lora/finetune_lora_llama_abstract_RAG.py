import os
from dataclasses import dataclass
import numpy as np
import torch
from datasets import load_dataset,DatasetDict
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType,PeftModel
import matplotlib.pyplot as plt
import pandas as pd
from huggingface_hub import login
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
import json
from dataclasses import dataclass, asdict

#from torch.amp import GradScaler
#scaler = GradScaler("cuda")
#source biolaysumm/bin/activate

@dataclass
class Config:
    output_dir: str = "output"
    checkpoint: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Update to LLaMA 3 checkpoint
    experiment_name: str = "LLaMA_RAG_lora_lr1e5_epo1_rank8_PLOS_0405"
    dataset_name: str = "BioLaySumm/BioLaySumm2025-PLOS"
    max_length: int = 2048
    optim_type: str = "adamw_torch"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4  
    per_device_eval_batch_size: int = 2
    n_epochs: int = 1
    freeze_layers: int = 20  # other option 16,20,24
    lr: float = 1e-5
    lora_r: int = 8
    lora_alpha: float = lora_r * 2
    lora_dropout: float = 0.1
    lora_bias: str = "none"
    def save(self, path: str):
          with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)
    @staticmethod
    def load(path: str):
        with open(path, "r") as f:
            data = json.load(f)
        return Config(**data)



def rag_format_prompt(sample):
    prompt = f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are an expert science communicator. Your task is to generate a **clear, accurate, and formal** summary of biomedical research articles.
    The summary should be **accessible to a general audience** while maintaining scientific rigor.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Title: {sample['title']}  
    Abstract: {sample['abstract']}  

    Supporting Text:
    {sample['retrieved_context']}

    Provide a **formal summary** of the article in {summary_word_len}. **Do not include explanations, self-reflections, or additional notes.** 
    Keep the response strictly to the summary.The output should begin directly with the summary text itself.<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    {sample['summary']}<|eot_id|>
    """
    return {
        "input_text": prompt,  # Model input (including expected output)
    }

def summary_length():
    if config.dataset_name == "BioLaySumm/BioLaySumm2025-PLOS":
        return '100-300 words'
    if config.dataset_name == "BioLaySumm/BioLaySumm2025-eLife":
        return '200-600 words'

def extract_abstract(example):
    example["abstract"] = example["article"].split("\n")[0]  # Extract text before first newline, which is the abstract
    return example

def extract_main_text(example):
    # Split at the first newline character
    parts = example["article"].split("\n", 1)
    # Get everything after the first newline
    example["main_text"] = parts[1] if len(parts) > 1 else ""
    return example

def tokenize_data(sample):
    encodings = tokenizer(
        sample["input_text"], 
        truncation=True, 
        #padding="max_length", 
        padding="longest",
        max_length=config.max_length
    )
    input_ids = torch.tensor(encodings["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(encodings["attention_mask"], dtype=torch.long)
    # Ensure labels are the same as input_ids but mask padding tokens
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100 
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels  # Should now always be 1D
    }

def len_abstract():
    for split in ['train', 'validation', 'test']:
        print(split)
        token_lengths = [len(tokenizer(i['abstract'])['input_ids']) for i in dataset[split]]
        max_length = max(token_lengths)
        min_length = min(token_lengths)
        print('abstract max length',max_length)
        print('abstract min length',min_length)
        #check very short abstracts
        short_indices = [idx for idx, length in enumerate(token_lengths) if length < 30]
        if len(short_indices) > 0:
            print('abstracts with abstracts shorter than 30 tokens, need to check:',len(short_indices))
            print(short_indices) #may contain parsing issue 

def drop_indices(dataset, drop_dict):
    new_dataset = {}
    for split, indices in drop_dict.items():
        if indices:  # Only modify if there are indices to remove
            new_dataset[split] = dataset[split].select(
                [i for i in range(len(dataset[split])) if i not in indices[0]]
            )
        else:  # If no indices to drop, keep the split unchanged
            new_dataset[split] = dataset[split]
    return DatasetDict(new_dataset)


def plot_training_and_validation_loss(history):
    history = trainer.state.log_history
    train_epochs = []
    train_losses = []
    val_epochs = []
    val_losses = []
    for entry in history:
        if "loss" in entry and "epoch" in entry:  # Training loss
            train_losses.append(entry["loss"])
            train_epochs.append(entry["epoch"])
        if "eval_loss" in entry and "epoch" in entry:  # Validation loss
            val_losses.append(entry["eval_loss"])
            val_epochs.append(entry["epoch"])
    if not train_epochs and not val_epochs:
        print("No loss history found.")
        return
    plt.figure(figsize=(8, 5))
    if train_epochs:
        plt.plot(train_epochs, train_losses, marker='o', linestyle='-', label="Training Loss")
    if val_epochs:
        plt.plot(val_epochs, val_losses, marker='s', linestyle='--', label="Validation Loss", color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig('./figures/%s_loss_epoch'%(config.experiment_name))  # Save plot without showing
    plt.close() 
    df = pd.DataFrame(history)
    df.to_csv('./figures/%s_loss_history.csv'%(config.experiment_name), index=False)


def add_chunks_field(example):
    text = example.get("main_text", "")
    if not text:
        return {
            "main_text_chunks": []}
    chunks = text_splitter.split_text(text)
    return {
        "main_text_chunks": chunks
    }



def retrieve_relevant_chunks(example, k=5):
    chunks = example.get("main_text_chunks", [])
    query = example.get("abstract", "")
    # If no chunks or abstract, return empty retrieved context
    if not chunks or not query:
        example["retrieved_context"] = ""
        return example
    # Build FAISS index for the chunks
    docsearch = FAISS.from_texts(chunks, embedder)
    # Retrieve top-k relevant chunks using abstract as query
    retrieved_docs = docsearch.similarity_search(query, k=k)
    retrieved_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    example["retrieved_context"] = retrieved_context
    return example


if __name__ == "__main__":
    config = Config()
    config.save("./configfile/finetune_%s_config.json"%(config.experiment_name)) 
    training_args = TrainingArguments(
        output_dir="output",
        overwrite_output_dir=True,
        report_to="none",
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=10,
        optim=config.optim_type,
        fp16=True,
        #bf16=True,# for A100 and H100
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
        layers_to_transform=[i for i in range(32) if i >= config.freeze_layers],  # Apply LoRA to higher layers 
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

    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 


    ##load data
    dataset = load_dataset(config.dataset_name)
    #dataset['train'][0]['article']
    #dataset['train'][0]['summary']

    dataset = dataset.map(extract_abstract)
    dataset = dataset.map(extract_main_text)
    dataset.column_names

    if config.dataset_name == "BioLaySumm/BioLaySumm2025-PLOS":
        plos_drop_dict={'train':[[725, 1939, 4226, 4842, 5991, 6310, 12050, 13498, 14104, 14199, 18921, 21808, 22922]],'validation':[],'test':[]} # drop due to inccorrect abstract
        dataset = drop_indices(dataset, plos_drop_dict)

    #len_abstract()


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # n chararcters
        chunk_overlap=50,    
        separators=["\n\n", "\n", ".", "?", "!", " ", ""])

    chunked_dataset = dataset.map(add_chunks_field,batched=False) 

    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    #chunked_dataset = chunked_dataset.map(retrieve_relevant_chunks)
    chunked_dataset["train"] = chunked_dataset["train"].map(retrieve_relevant_chunks)
    chunked_dataset["validation"] = chunked_dataset["validation"].map(retrieve_relevant_chunks)


    #test_train = dataset["train"].select(range(50))  # First n samples
    #test_val = dataset["validation"].select(range(50))  # First n validation samples
    train_set =chunked_dataset["train"]
    val_set=chunked_dataset["validation"]


    summary_word_len = summary_length()
    formatted_train = train_set.map(rag_format_prompt, remove_columns=dataset["train"].column_names)
    formatted_val = val_set.map(rag_format_prompt, remove_columns=dataset["validation"].column_names)



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

    history = trainer.train()
    plot_training_and_validation_loss(history)


    #trainer.args.num_train_epochs += 2  # Increase epoch count
    #trainer.train(resume_from_checkpoint=True)  # Resume training


    #peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert bytes to GB
    #print(f"Peak GPU memory usage: {peak_memory:.2f} GB")

    login(token="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")
    model.push_to_hub("linf545/%s"%(config.experiment_name))

        #test loading the model
        #model = PeftModel.from_pretrained(config.checkpoint, "linf545/%s"%(config.experiment_name))


