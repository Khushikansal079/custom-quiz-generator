from transformers import BartTokenizer, BartForConditionalGeneration, TrainingArguments, Trainer
import pandas as pd
from datasets import Dataset, Features, Value
import evaluate
import nltk
import json
import os
import random

nltk.download('punkt')

# === CONFIGURATION ===
train_file = r"C:/Users/aditi/OneDrive/Desktop/train_v0.2 QuaC.json"
model_name = "voidful/bart-eqg-question-generator"
output_dir = "./bart-eqg-finetuned-500"

# === FILE CHECK ===
if not os.path.exists(train_file):
    raise FileNotFoundError(f"File not found at: {train_file}")

# === LOAD DATA ===
with open(train_file, 'r', encoding='utf-8') as f:
    quac_data = json.load(f)

# === EXTRACT 500 Q&A PAIRS ===
data = []
for item in quac_data.get("data", []):
    for paragraph in item.get("paragraphs", []):
        context = paragraph.get("context", "")
        for qa in paragraph.get("qas", []):
            question = qa.get("question", "")
            answer = qa.get("answers", [{}])[0].get("text", "") if qa.get("answers") else ""
            if context and question and answer:
                data.append({"context": context, "question": question, "answer": answer})

random.seed(42)
random.shuffle(data)
data = data[:500]

# === CREATE DATASET ===
df = pd.DataFrame(data)[["context", "question", "answer"]]
features = Features({
    "context": Value("string"),
    "question": Value("string"),
    "answer": Value("string")
})
dataset = Dataset.from_pandas(df, features=features)
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")

# === LOAD MODEL AND TOKENIZER ===
try:
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
except Exception as e:
    raise RuntimeError(f"Could not load model or tokenizer: {e}")

# === PREPROCESS FUNCTION ===
def preprocess(example):
    input_text = example['context']
    target_text = example['question']
    model_inputs = tokenizer(input_text, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(target_text, max_length=64, truncation=True, padding="max_length")["input_ids"]
    model_inputs["labels"] = labels
    return model_inputs

tokenized_train_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names, batched=True)
tokenized_eval_dataset = eval_dataset.map(preprocess, remove_columns=eval_dataset.column_names, batched=True)

# === METRIC COMPUTATION ===
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")

    bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels)

    return {
        "bleu": bleu_score["bleu"],
        "rouge1": rouge_score["rouge1"],
        "rougeL": rouge_score["rougeL"]
    }

# === TRAINING ARGS === (no evaluation_strategy used)
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    save_strategy="epoch",
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=10,
    fp16=False,
    report_to="none"
)

# === TRAINER ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    compute_metrics=compute_metrics
)

# === TRAIN & EVALUATE ===
print("Fine-tuning started...")
#trainer.train()
trainer.train(resume_from_checkpoint=True)

print("Running final evaluation...")
results = trainer.evaluate()
print("Final Evaluation Results:")
for metric, score in results.items():
    print(f"  {metric}: {score}")

# === SAVE MODEL ===
model.save_pretrained(os.path.join(output_dir, "final"))
tokenizer.save_pretrained(os.path.join(output_dir, "final"))
print("Fine-tuned model and tokenizer saved!")

