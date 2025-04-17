
from dataclasses import dataclass
import numpy as np
import torch
import transformers
from transformers import (
    pipeline
)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, pipeline

from datasets import Dataset
from datasets import load_dataset
from huggingface_hub import login
import re
from datasets import DatasetDict
import matplotlib.pyplot as plt
import statistics

# check toecker length of all data
def len_article_details():
    for split in ['train', 'validation', 'test']:
        print(split)
        token_lengths = [len(tokenizer(i['article'])['input_ids']) for i in dataset[split]]
        max_length = max(token_lengths)
        min_length = min(token_lengths)
        mean_length = statistics.mean(token_lengths)
        median_length = statistics.median(token_lengths)
        print('article max length',max_length)
        print('article min length',min_length)
        print('Article mean length:', mean_length)
        print('Article median length:', median_length)
        # llama3 context window 128k tokens
        # gemma2 context length of 8192 tokens
        if max_length > 128000:
            num_exceeding_128k = sum(1 for length in token_lengths if length > 128000)
            print('number of articles exceeding 128k:',num_exceeding_128k)
        short_indices = [idx for idx, length in enumerate(token_lengths) if length < 500]
        if len(short_indices) > 0:
            print('articles with less than 30 tokens, need to check:',len(short_indices))
            print(short_indices)
        
def len_abstract():
    for split in ['train', 'validation', 'test']:
        print(split)
        token_lengths = [len(tokenizer(i['abstract'])['input_ids']) for i in dataset[split]]
        max_length = max(token_lengths)
        min_length = min(token_lengths)
        mean_length = statistics.mean(token_lengths)
        median_length = statistics.median(token_lengths)
        print('abstract max length',max_length)
        print('abstract min length',min_length)
        print('Article mean length:', mean_length)
        print('Article median length:', median_length)
        #check very short abstracts
        short_indices = [idx for idx, length in enumerate(token_lengths) if length < 30]
        if len(short_indices) > 0:
            print('abstracts with abstracts shorter than 30 tokens, need to check:',len(short_indices))
            print(short_indices) #may contain parsing issue 


def extract_abstract(example):
    example["abstract"] = example["article"].split("\n")[0]  # Extract text before first newline, which is the abstract
    return example



def drop_indices(dataset, drop_dict):
    """
    Removes specified indices from each dataset split in the dataset dictionary.

    Parameters:
        dataset (DatasetDict): The dataset dictionary containing 'train', 'validation', and 'test'.
        drop_dict (dict): A dictionary specifying which indices to remove from each split.

    Returns:
        DatasetDict: A new dataset dictionary with specified indices removed.
    """
    new_dataset = {}
    for split, indices in drop_dict.items():
        if indices:  # Only modify if there are indices to remove
            new_dataset[split] = dataset[split].select(
                [i for i in range(len(dataset[split])) if i not in indices[0]]
            )
        else:  # If no indices to drop, keep the split unchanged
            new_dataset[split] = dataset[split]
    return DatasetDict(new_dataset)


def count_summary_word_lengths_by_split(dataset):
    word_counts_by_split = {}
    for split in ['train', 'validation']: # test set does not have summary
        split_counts = []
        for summary in dataset[split]['summary']:
            if summary:
                split_counts.append(len(summary.split()))
        word_counts_by_split[split] = split_counts
    return word_counts_by_split

def plot_summary_word_distribution_by_split(word_counts_by_split,name):
    plt.figure(figsize=(10, 6))
    for split, counts in word_counts_by_split.items():
        plt.hist(counts, bins=50, alpha=0.5, label=split, edgecolor='black', linewidth=0.5)
    plt.title('Summary Word Count Distribution by Dataset Split')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./figures/EDA/%s_summary_word_count_distribution.png'%(name))
    plt.close()



model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
config = AutoConfig.from_pretrained(model_id)
config.rope_scaling = {"type": "linear", "factor": 2.0}  
model = AutoModelForCausalLM.from_pretrained(model_id, config=config, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)



dataset = load_dataset("BioLaySumm/BioLaySumm2025-eLife")
dataset = dataset.map(extract_abstract)
dataset.column_names

len_article_details()
len_abstract()
word_counts_by_split = count_summary_word_lengths_by_split(dataset)
plot_summary_word_distribution_by_split(word_counts_by_split,"eLife")

dataset = load_dataset("BioLaySumm/BioLaySumm2025-PLOS")
dataset = dataset.map(extract_abstract)
dataset.column_names

len_article_details()
len_abstract()

plos_drop_dict={'train':[[725, 1939, 4226, 4842, 5991, 6310, 12050, 13498, 14104, 14199, 18921, 21808, 22922]],'validation':[],'test':[]} # drop due to inccorrect abstract
dataset = drop_indices(dataset, plos_drop_dict)
len_abstract()
word_counts_by_split = count_summary_word_lengths_by_split(dataset)
plot_summary_word_distribution_by_split(word_counts_by_split,"plos")

prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a skilled science communicator. Your task is to generate a plain-language summary of biomedical research articles, making them accessible to a general audience without specialized knowledge.

<|start_header_id|>user<|end_header_id|>

Generate a plain-language summary for the following biomedical research article, ensuring clarity, conciseness, and accessibility to a non-expert audience.

Title: {title}

Full Text: {article}

<|start_header_id|>assistant<|end_header_id|>
""".format(title=dataset['train'][0]['title'], article=dataset['train'][0]['article'])

#inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to("cpu") # CPU for testing purpose only
inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to("cuda")
summary_ids = model.generate(**inputs, max_new_tokens=400, do_sample=True)
output_text= tokenizer.decode(summary_ids[0], skip_special_tokens=True)
match = re.search(r"assistant\nHere is a plain-language summary of the article:\n\n(.+)", output_text, re.DOTALL)
if match:
    cleaned_summary = match.group(1).strip()
else:
    cleaned_summary = "No summary found."

print(cleaned_summary)

# Initialize pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

example=dataset['train'][0]['article']

# Run test prompt
print(pipe("Summarize this biomedical research paper:", min_length=200, max_length=400))