
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

os.environ["OPENAI_API_KEY"] ='XXXXXXXXXXXXXx'


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
            entry[f"{metric.name}_reason"] = getattr(metric, 'reason', "N/A")
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


correctness_metric_old= GEval(
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


# 'normal_summaries','bad_factuality_summaries','bad_readability_summaries','bad_relavance_summaries','bad_summaries','paraphrase_summaries'

for f in ['normal_summaries','bad_summaries']:
    generated_df=pd.read_parquet("./output/synthesized_data/%s.parquet"%(f))
    generated_df['Generated_LaySummary'] = generated_df['summary'].apply(extract_lay_summary)
    test_cases = []
    for i in range(20):  # First 10 test cases
        test_case = LLMTestCase(
            input=dataset['validation'][i]['abstract'],
            actual_output=generated_df['Generated_LaySummary'][i],
            expected_output=dataset['validation'][i]['summary'],)
        test_cases.append(test_case)
    results = evaluate(test_cases=test_cases, metrics=[correctness_metric])
    # Parse the results
    parsed_results = parse_results(results)
    # Save as CSV file
    csv_file = "./output/evaluation_results/20250401_synthesized_data_eval/v2_plos_synthesized_evaluation_results_%s.csv"%(f)
    with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=parsed_results[0].keys())
        writer.writeheader()
        writer.writerows(parsed_results)
