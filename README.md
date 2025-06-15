# 🧬 BioLaySummarization

BioLaySummarization is a deep learning-based project designed to generate **lay summaries** for biomedical research papers. It leverages NLP models to convert complex biomedical content into simpler, layperson-friendly explanations.

## 📦 Installation

### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/your-username/BioLaySummarization.git
cd BioLaySummarization
```

### 2️⃣ **Create a Virtual Environment (Recommended)**
To keep dependencies isolated, it's best to use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## 🚀 **Run the Project**


### **Finetune the Model**

```bash
python ./lora/finetune_lora_llama_abstract_RAG.py 
```
Settings can be adjusted in the configuration class defined in the .py file

### **Evaluate the Model**
We used the evaluation script provided in the following repository to assess performance on the validation set: https://github.com/gowitheflow-1998/BioLaySumm2025.git

### **Run Inference**
To generate lay summaries for new biomedical papers:
```bash
python ./inference/inference_lora_RAG_local_knowledge.py
```
Settings can be adjusted in the configuration class defined in the .py file

---

## 🛠 **Project Structure**
```
├── EDA/                        # Summary statistics and exploratory data analysis of the dataset  
├── configfile/                # Configuration files for specific experiments  
├── models/                    # Saved model checkpoints  
├── evaluation/                # Scripts for evaluation on the validation set  
├── figures/                   # Figures used for publication and intermediate visualizations  
├── inference/                 # Scripts for model inference  
├── lora/                      # Scripts for LoRA fine-tuning  
├── output/                    # Output directory for experiment results  
│   ├── evaluation_results/           # Evaluation results using G-Eval  
│   ├── generated_summaries/          # Generated summaries from various experiments  
│   ├── synthesized_data/             # Synthesized data used in G-Eval assessments  
│   ├── validationset_evaluation/     # Performance metrics on the validation set  
│   └── aggregated_scores.csv         # Aggregated validation performance scores across experiments  
├── biolaysumm2025_exp_design.xlsx   # Index of experiments and configuration settings  
├── requirements.txt           # Project dependencies  
├── README.md                   
```

---

## 📌 **Troubleshooting**

If you encounter issues:  

1. **Ensure Python 3.10+ is installed**:
   ```bash
   python3 --version
   ```
2. **Reinstall dependencies**:
   ```bash
   pip install --upgrade --force-reinstall -r requirements.txt
   ```
3. **Check if all required libraries are installed**:
   ```bash
   pip list | grep -E "torch|transformers|numpy"
   ```

---

## 🤝 **Contributing**
We welcome contributions! Feel free to submit a pull request or open an issue.

---

## 📜 **License**
This project is licensed under the MIT License.

