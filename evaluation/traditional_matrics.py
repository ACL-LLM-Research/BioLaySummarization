import os, sys, json
import textstat
import numpy as np
from rouge_score import rouge_scorer
from bert_score import score
import torch
import pandas as pd
from datasets import load_dataset
import nltk
nltk.download('punkt_tab')


def calc_rouge(preds, refs):
  scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], \
                                    use_stemmer=True, split_summaries=True)
  scores = [scorer.score(p, refs[i]) for i, p in enumerate(preds)]
  return np.mean([s['rouge1'].fmeasure for s in scores]), \
         np.mean([s['rouge2'].fmeasure for s in scores]), \
         np.mean([s['rougeLsum'].fmeasure for s in scores])

def calc_bertscore(preds, refs):
  P, R, F1 = score(preds, refs, lang="en", verbose=True, device='cuda:0')
  return np.mean(F1.tolist())

def calc_readability(preds):
  fkgl_scores = []
  cli_scores = []
  dcrs_scores = []
  for pred in preds:
    fkgl_scores.append(textstat.flesch_kincaid_grade(pred))
    cli_scores.append(textstat.coleman_liau_index(pred))
    dcrs_scores.append(textstat.dale_chall_readability_score(pred))
  return np.mean(fkgl_scores), np.mean(cli_scores), np.mean(dcrs_scores)



def evaluate(generated_list, ref_list):
    score_dict = {}
    # Relevance scores
    print("Calculating ROUGE...")
    rouge1_score, rouge2_score, rougel_score = calc_rouge(generated_list, ref_list)
    score_dict['ROUGE1'] = rouge1_score
    score_dict['ROUGE2'] = rouge2_score
    score_dict['ROUGEL'] = rougel_score
    print("Calculating BERTScore...")
    bert_score = calc_bertscore(generated_list, ref_list)
    score_dict['BERTScore'] = bert_score
    # Readability scores
    print("Calculating Readability metrics...")
    fkgl_score, cli_score, dcrs_score = calc_readability(generated_list)
    score_dict['FKGL'] = fkgl_score  # Flesch-Kincaid Grade Level
    score_dict['CLI'] = cli_score    # Coleman-Liau Index
    score_dict['DCRS'] = dcrs_score  # Dale-Chall Readability Score
    return score_dict


dataset = load_dataset('BioLaySumm/BioLaySumm2025-PLOS')
val_set=dataset["validation"]
ref_summ=val_set['summary']

all_results = []

for i in ['LLaMA_base/PLOS_val_summaries','llama_RAG_local_knowledge/PLOS_val_summaries','LLaMA_lora/PLOS_val_summaries','llama_RAG_local_knowledge/PLOS_val_summaries']:
    generated_df=pd.read_parquet('./output/generated_summaries/%s.parquet'%(i))
    generated_summ=generated_df['summary'].to_list()
    results = evaluate(generated_summ, ref_summ)
    results['Experiment'] = i # Add model name to track the source
    all_results.append(results)


results_df = pd.DataFrame(all_results)
results_df.to_csv('./output/evaluation_summary/evaluation_summary20250403.csv', index=False)