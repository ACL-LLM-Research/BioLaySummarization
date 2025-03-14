



import os
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from datasets import load_dataset
import csv
import json


def extract_abstract(example):
    example["abstract"] = example["article"].split("\n")[0]  # Extract text before first newline, which is the abstract
    return example

os.environ["OPENAI_API_KEY"] ='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'

correctness_metric = GEval(
    #name="BioSumm",
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

test_case = LLMTestCase(
    input=dataset['train'][0]['abstract'],
    actual_output='''The kidneys develop from intermediate mesoderm (IM) during embryonic development. The proneph, mesoneph, and meteph are three kidneys that form sequentially from IM. The proneph and mesph degenerate, while meteph serves as adult kidney. Lower vertebrates like fish and amphibians develop prone during embryonic stages and form mesph as adult. Each kidney contains neph as its basic unit.
Zebraf is an ideal model for studying kidney development. The kidney has two neph as opposed to thousands in mammalian meteph. During development, prone neph undergoes significant morphogenesis, including mid migration of pod and extension of tubules. The prone is composed of proximal and dist segments. 
Studies suggest that zraf prone has eight regions, including proximal and dist. Proximal tubule recovers solutes, while dist tub transports. The prone has regional boundaries of solute transporters. Some solutes are expressed in proximal or dist regions. Slc1, Sl2, and Sl3 are proximal markers while Sl4 and Sl5 are dist markers.
Retino acid (RA) is essential for kidney development. RA is produced by upper trunk during gastrulation and early somogenesis. RA induces proximal''',
    expected_output=dataset['train'][0]['summary'],
)



results =evaluate(test_cases=[test_case], metrics=[correctness_metric])



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

# Parse the results
parsed_results = parse_results(results)

# Save as CSV file
csv_file = "evaluation_results.csv"
with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=parsed_results[0].keys())
    writer.writeheader()
    writer.writerows(parsed_results)
