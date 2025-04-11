
import os
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from datasets import load_dataset
import csv
import json
import pandas as pd
import re
import sys

#usage python ./evaluation/LLM_evaluator_valset20.py LLaMA_base PLOS_val_summaries

os.environ["OPENAI_API_KEY"] ='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'

def extract_abstract(example):
    example["abstract"] = example["article"].split("\n")[0]  # Extract text before first newline, which is the abstract
    return example


def parse_results(results):
    parsed_data = []
    for test_result in results.test_results:
        entry = {
            "test_name": test_result.name,
            "input_text": test_result.input,
            "actual_output": test_result.actual_output,
            "expected_output": test_result.expected_output,
            "success": test_result.success,
        }
        # Extract metric scores
        for metric in test_result.metrics_data:
            entry[f"{metric.name}_score"] = metric.score
            entry[f"{metric.name}_reason"] = metric.reason
        parsed_data.append(entry)
    return parsed_data


def extract_lay_summary(text):
    # Use regex to find the section that starts with "Lay Summary:" and ends before the "Keywords:"
    match = re.search(r'assistant\s*\n(.*)', text, re.DOTALL)
    # If a match is found, return the Lay Summary content
    if match:
        return match.group(1).strip()
    else:
        return None 


correctness_metric = GEval(
    name="BioSumm",
    criteria="""
    Evaluate the generated lay summary on the following three criteria:
    1. **Relevance (1-5)**: Does the summary retain all major findings and themes of the source abstract? Score higher if it covers key points, even if phrased differently. Penalize only if essential information is missing or incorrect topics are introduced.
    2. **Readability (1-5)**: Is the summary easy to understand for a non-expert audience? Consider fluency, sentence structure, and clarity. Avoid penalizing for simplified language unless it introduces confusion.
    3. **Factuality (1-5)**: Does the summary accurately reflect the scientific claims in the source abstract? Check for hallucinations or misinterpretations, not just omissions.
    Each criterion should be scored from 1 (poor) to 5 (excellent). Then provide a final **Overall Score
    """,
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    model="gpt-3.5-turbo" ,
    #model="gpt-4-turbo" 
)

dataset = load_dataset("BioLaySumm/BioLaySumm2025-PLOS")
dataset = dataset.map(extract_abstract)


# e.g.  argv[1]=LLaMA_base argv[2]=PLOS_val_summaries
experiment_name =sys.argv[1] 
generated_text_file=sys.argv[2] 

generated_df=pd.read_parquet("./output/generated_summaries/%s/%s.parquet"%(experiment_name,generated_text_file))
#generated_df['Generated_LaySummary'] = generated_df['summary'].apply(extract_lay_summary)
generated_df.rename(columns={'summary': 'Generated_LaySummary'}, inplace=True)
test_cases = []
for i in range(20):  # to be replace with a list of random indices
    test_case = LLMTestCase(
        input=dataset['validation'][i]['abstract'],
        actual_output=generated_df['Generated_LaySummary'][i],
        expected_output=dataset['validation'][i]['summary'],)
    test_cases.append(test_case)

results = evaluate(test_cases=test_cases, metrics=[correctness_metric])
# Parse the results
parsed_results = parse_results(results)
# Save as CSV file
csv_file = "./output/evaluation_results/Gval_results_val/%s.csv"%(experiment_name+'_'+generated_text_file)
with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=parsed_results[0].keys())
    writer.writeheader()
    writer.writerows(parsed_results)
