



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
    match = re.search(r'Lay Summary:\s*(.*?)\s*Keywords:', text, re.DOTALL)
    # If a match is found, return the Lay Summary content
    if match:
        return match.group(1).strip()
    else:
        return None 
    

correctness_metric = GEval(
    name="BioSumm",
    criteria="""
    Evaluate the generated lay summary based on the following criteria:
    1. **Relevance** (Accuracy of Content & Semantic Overlap):  
    2. **Readability** (Comprehensibility & Fluency, ease of understanding):  
    3. **Factuality** (Correctness & Faithfulness to Source):  
    Assign a score for each criterion (1-5) and provide a final overall score.
    """,
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    model="gpt-3.5-turbo" ,
    #model="gpt-4-turbo" 
)

dataset = load_dataset("BioLaySumm/BioLaySumm2025-PLOS")
dataset = dataset.map(extract_abstract)

generated_df=pd.read_csv("./output/generated_summaries/20250316_doubao_test/plos_lay_summaries.csv")
subset_df=generated_df.iloc[:1376]# only validation set
subset_df['Generated_LaySummary'] = subset_df['Generated Lay Summary'].apply(extract_lay_summary)


test_cases = []
for i in range(10):  # First 10 test cases
    test_case = LLMTestCase(
        input=dataset['validation'][i]['abstract'],
        actual_output=subset_df['Generated_LaySummary'][i],
        expected_output=dataset['validation'][i]['summary'],)
    test_cases.append(test_case)

results = evaluate(test_cases=test_cases, metrics=[correctness_metric])

# Parse the results
parsed_results = parse_results(results)

# Save as CSV file
csv_file = "plos_evaluation_results_first_10_cases.csv"
with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=parsed_results[0].keys())
    writer.writeheader()
    writer.writerows(parsed_results)
