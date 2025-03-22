from datasets import load_dataset
from openai import OpenAI
import pandas as pd
import csv

client = OpenAI(api_key="XXXXXXXX", base_url="https://api.deepseek.com")

dataset = load_dataset("BioLaySumm/BioLaySumm2025-eLife")

print("Dataset Splits:", dataset.keys())
for split in dataset.keys():
    print(f"Total records in '{split}':", len(dataset[split]))

# system prompt
system_prompt = """You are a highly advanced scientific communication assistant specializing in summarizing biomedical research articles into clear, concise, and engaging lay summaries.  
Your primary goal is to make complex scientific findings accessible to non-expert audiences, including medical practitioners, interdisciplinary researchers, and the general public.  
Your summaries must retain the key findings and scientific integrity of the research while avoiding technical jargon and overly complex language.  
Your output should be factually accurate, well-structured, and easy to understand, ensuring that readers can grasp the main points without requiring prior domain knowledge."""

# user prompt 模板
user_prompt_template = """Please summarize the following biomedical research article into a clear, lay-friendly summary following these guidelines:

- Introduction: Brief background and why it's important  
- Research Objective  
- Methodology  
- Key Findings  
- Implications  
- Conclusion

Language Guidelines:
- Use simple, clear language for a general audience  
- Avoid technical jargon  
- Write at a Flesch-Kincaid Grade Level of 8 or below  
- Be neutral and informative  
- Do not use bullet points in the output  

Output Format:
Lay Summary: Present the summary in paragraph format, with a total length of 200–300 words. Do not use section titles, bullet points, or any structural formatting. Do not mention word count or include any formatting-related instructions in the output. Avoid including any information unrelated to the summary itself.

---
Title: {title}  
Year: {year}  
Section Headings: {section_headings}  
Keywords: {keywords}  
Full Article Content: {article}  
"""

output_data = []
output_csv = "elife_lay_summaries.csv"

with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Global Index", "Split", "Split Index", "Title", "Year", "Keywords", "Generated Lay Summary"])

    global_index = 0

    for split in ["train", "validation", "test"]:
        for split_index, row in enumerate(dataset[split]):
            if global_index >= 100:  # Limit to 100 records for testing
                break

            user_input = {
                "title": str(row.get("title", "N/A")),
                "article": str(row.get("article", "N/A")),
                "section_headings": str(row.get("section_headings", "N/A")),
                "keywords": str(row.get("keywords", "N/A")),
                "year": str(row.get("year", "N/A")),
            }

            user_prompt = user_prompt_template.format(**user_input)

            try:
                response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    stream=False
                )
                output = response.choices[0].message.content.strip()
            except Exception as e:
                output = f"[Error] {str(e)}"
                print(f"Error on {split} index {split_index}: {e}")

            print(output)

            output_row = {
                "Global Index": global_index,
                "Split": split,
                "Split Index": split_index,
                "Title": user_input["title"],
                "Year": user_input["year"],
                "Keywords": user_input["keywords"],
                "Generated Lay Summary": output
            }

            output_data.append(output_row)
            writer.writerow(output_row.values())

            global_index += 1

df = pd.DataFrame(output_data)[[
    "Global Index", "Split", "Split Index", "Title", "Year", "Keywords", "Generated Lay Summary"
]]
df.to_excel("elife_lay_summaries.xlsx", index=False)

print(f"Lay summaries saved to {output_csv} and elife_lay_summaries.xlsx")
