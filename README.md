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

### **Train the Model**
```bash
python train.py --dataset data/bio_papers.json --epochs 5
```
This will train the model using the provided biomedical dataset and save the best model checkpoint.

### **Finetune the Model**


```bash
python ./lora/finetune-lora.py --lora_alpha 32 --lora_r 16
```

### **Evaluate the Model**
```bash
python evaluate.py --model models/best_model.pth --dataset data/bio_test.json
```
This will load the trained model and evaluate its performance on a test dataset.

### **Run Inference**
To generate lay summaries for new biomedical papers:
```bash
python predict.py --input sample_paper.txt
```

---

## 🛠 **Project Structure**
```
├── data/               # Dataset files (raw & processed)
├── models/             # Saved model checkpoints
├── src/
│   ├── preprocess.py   # Data preprocessing functions
│   ├── train.py        # Model training script
│   ├── evaluate.py     # Model evaluation script
│   ├── predict.py      # Inference script for summarization
├── notebooks/          # Jupyter notebooks for experiments
├── requirements.txt    # Required dependencies
├── README.md           # Project documentation
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

