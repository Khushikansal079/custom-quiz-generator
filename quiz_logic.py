# quiz_logic.py
import random
import nltk
from transformers import pipeline
from nltk.tokenize import sent_tokenize

# Download required tokenizer
nltk.download('punkt', quiet=True)

# Load NLI model
nli = pipeline("text-classification", model="facebook/bart-large-mnli")

def validate_inputs(context, num_questions, difficulty):
    if not context.strip():
        return False, "Context cannot be empty."
    sentences = sent_tokenize(context)
    if len(sentences) < num_questions:
        return False, f"Context has only {len(sentences)} sentences, but {num_questions} questions requested."
    if difficulty not in ["easy", "medium", "hard"]:
        return False, "Difficulty must be 'easy', 'medium', or 'hard'."
    return True, sentences

def apply_noise(sentence: str, level: str) -> str:
    if level == "easy":
        return sentence
    elif level == "medium":
        if "Sun" in sentence:
            return sentence.replace("Sun", "Moon")
        return sentence.replace("is", "is not") if "is" in sentence else sentence
    elif level == "hard":
        if "eight" in sentence:
            return sentence.replace("eight", "ten")
        return sentence.replace("planets", "stars") if "planets" in sentence else sentence
    return sentence

def generate_statements(context, n, difficulty, sentences):
    random.seed(42)
    selected = random.sample(sentences, min(n * 2, len(sentences)))
    final = []
    for s in selected:
        clean = s.strip()
        modified = apply_noise(clean, difficulty)
        label = "ENTAILMENT" if clean == modified else "CONTRADICTION"
        final.append({"statement": modified, "actual_label": label})
        if len(final) >= n:
            break
    return final

def score_answers(context, answers):
    score = 0
    results = []
    for answer in answers:
        statement = answer.get('statement')
        user_answer = answer.get('user_answer', '').strip().lower()
        if user_answer not in ['true', 'false']:
            results.append({
                "statement": statement,
                "result": "Invalid answer. Please use 'true' or 'false'."
            })
            continue
        input_text = f"{context} [SEP] {statement}"
        result = nli(input_text)[0]
        if result["label"] == "neutral":
            results.append({
                "statement": statement,
                "result": "Skipped due to ambiguous statement."
            })
            continue
        model_label = "ENTAILMENT" if result["label"] == "entailment" else "CONTRADICTION"
        is_correct = (model_label == "ENTAILMENT" and user_answer == "true") or \
                     (model_label == "CONTRADICTION" and user_answer == "false")
        results.append({
            "statement": statement,
            "result": "Correct" if is_correct else f"Incorrect (Correct answer: {'True' if model_label == 'ENTAILMENT' else 'False'})"
        })
        if is_correct:
            score += 1
    return score, results