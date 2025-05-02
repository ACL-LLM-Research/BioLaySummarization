import os, gc, torch, json
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from datasets import load_dataset,DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from dataclasses import dataclass, asdict
#from langchain_community.retrievers import PubMedRetriever
#retriever = PubMedRetriever(top_k=3 retmode="json")


@dataclass
class Config:
    output_dir: str = "output"
    checkpoint: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Update to LLaMA 3 checkpoint
    experiment_index: str = '2'
    #experiment_name: str = "RAG_main_text_general_retraiever_2"
    #dataset_name: str = "BioLaySumm/BioLaySumm2025-PLOS"
    max_new_tokens: int= 800
    num_beams: int= 4
    input_max_length: int = 100000# using 100000 if using large RAG like exp 10 and 11
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


def extract_abstract(example):
    example["abstract"] = example["article"].split("\n")[0]  # Extract text before first newline, which is the abstract
    return example

def extract_main_text(example):
    # Split at the first newline character
    parts = example["article"].split("\n", 1)
    # Get everything after the first newline
    example["main_text"] = parts[1] if len(parts) > 1 else ""
    return example

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


def rag_format_inference_prompt(sample):
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are an expert science communicator. Your task is to generate a **clear, accurate, and formal** summary of biomedical research articles.
    The summary should be **accessible to a general audience** while maintaining scientific rigor.<|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Title: {sample['title']}
    Abstract: {sample['abstract']}

    Supporting Text:
    {sample['retrieved_context']}

    Provide a **formal summary** of the article in {summary_word_len}. **Do not include explanations, self-reflections, preamble, extra formatting, or additional notes.** 
    Keep the response strictly to the summary. The output should begin directly with the summary text itself.<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
    return {
        "input_text": prompt,  # Model input (including expected output)
    }

def generate_output(sample):
    inputs = tokenizer(sample["input_text"], return_tensors="pt",  truncation=True,max_length=config.input_max_length)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device) 
    output_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=config.max_new_tokens,
                                num_beams=config.num_beams,do_sample=False,
                                temperature=None,top_p=None, pad_token_id=tokenizer.eos_token_id)
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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # n chararcters
        chunk_overlap=50,    
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )

    autoconfig = AutoConfig.from_pretrained(config.checkpoint)
    autoconfig .rope_scaling = {"type": "linear", "factor": 2.0}  
    model = AutoModelForCausalLM.from_pretrained(config.checkpoint, config=autoconfig, 
                                                device_map="auto",
                                                torch_dtype=torch.bfloat16) # bf16 if using H100 
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    for j in ["BioLaySumm/BioLaySumm2025-PLOS", "BioLaySumm/BioLaySumm2025-eLife"]:
        dataset = load_dataset(j)
        #dataset = dataset.select(range(10)) # test only
        dataset = dataset.map(extract_abstract)
        dataset = dataset.map(extract_main_text)
        dataset.column_names

        if j == "BioLaySumm/BioLaySumm2025-PLOS":
            plos_drop_dict={'train':[[725, 1939, 4226, 4842, 5991, 6310, 12050, 13498, 14104, 14199, 18921, 21808, 22922]],'validation':[],'test':[]} # drop due to inccorrect abstract
            dataset = drop_indices(dataset, plos_drop_dict)


        for i in ['test', 'validation']:
            selected_set = dataset[i]
            chunked_dataset = selected_set.map(add_chunks_field,batched=False)  
            embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            #chunked_dataset = chunked_dataset.map(retrieve_relevant_chunks)
            chunked_dataset = chunked_dataset.map(retrieve_relevant_chunks)
            val_set=chunked_dataset

            summary_word_len = summary_length(j)
            formatted_val = val_set.map(rag_format_inference_prompt, remove_columns=dataset[i].column_names)
            print('start to generate output')
            generated_val = formatted_val.map(generate_output)
            print('writing outputs')
            output_path = "./output/generated_summaries/indexed_experiments/experiment%s/%s_%s_summaries.txt" % (config.experiment_index,j.split('-')[1], i)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                for line in generated_val['summary']:
                    f.write(line + "\n")
            generated_val.to_parquet("./output/generated_summaries/indexed_experiments/experiment%s/%s_%s_check.csv"%(config.experiment_index,j.split('-')[1],i))

            del embedder, chunked_dataset, formatted_val, generated_val, selected_set
            free_cuda()
    

    del model, tokenizer, text_splitter
    free_cuda()
