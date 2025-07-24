# 📚AI-Powered Custom Quiz Generator - QuizCraft Ai 

Generate personalized MCQs, short answer, and true/false questions using Hugging Face Transformers and a Streamlit UI.

## 💡 Features
- Question generator (MCQ, short answer, true/false)
- Streamlit-based frontend
- Cosine Similarity, BLEU-1, ROUGE -1 AND ROUGE-L Evaluation
- Fine-tuned FLAN-T5 integration
- Customization: Select topic, difficulty, and number of questions



## 🚀 How to Run

```bash
git clone https://github.com/YOUR_USERNAME/custom-quiz-generator.git
cd custom-quiz-generator

# (Optional) Create virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows
# or
source venv/bin/activate      # On Mac/Linux

# Install required dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

```
## Repo Struture
```
custom-quiz-generator/
│
├── app.py                          # Streamlit UI
├── fine_tune_and_evaluation.py     # Fine-tuning & evaluation script
├── flan_t5_finetuned_model/        # Directory storing the fine-tuned FLAN-T5 model
├── mcq_generator.py                # MCQ generation script                  
├── quiz_logic.py                   # Core quiz generation logic
├── short_answer_generator.py       # Script for short answer generation
├── truefalse_quiz.py               # True/False question generator
├── train_v0.2_QuaC.json            # Training dataset
├── outputs/                        # Stores generated questions/outputs
├── valhalla/                       # T5-based fine-tuned models
├── requirements.txt                # Project dependencies
├── FineTuneAndEvaluationscores.ipynb  # Evaluation notebook
├── README.md                       # Project documentation
└── .gitignore                      # Git ignore rules
```
