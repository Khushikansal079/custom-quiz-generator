import random
import nltk
from transformers import pipeline
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab', quiet=True)
# Load NLI model
nli = pipeline("text-classification", model="facebook/bart-large-mnli")

class generate_true_false:
    def __init__(self):
        pass
    def validate_inputs(self, context, num_questions, difficulty):
        if not context.strip():
            raise ValueError("Context cannot be empty.")
        sentences = sent_tokenize(context)
        return sentences

    def apply_noise(self, sentence: str, level: str) -> str:
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

    # Statement generator
    def generate_statements(self, context, n, difficulty, sentences):
        random.seed(42)
        selected = random.sample(sentences, min(n * 2, len(sentences)))
        final = []
        for s in selected:
            clean = s.strip()
            modified = self.apply_noise(clean, difficulty)
            label = "ENTAILMENT" if clean == modified else "CONTRADICTION"
            final.append((modified, label))
            if len(final) >= n:
                break
        return final

    # Get valid user answer
    def get_user_answer(self):
        while True:
            user = input("True or False? ").strip().lower()
            if user in ["true", "false"]:
                return user
            print("Please enter 'true' or 'false'.")

    # Main quiz logic
    def run_quiz(self, context, num_questions, difficulty):
        try:
            sentences = self.validate_inputs(context, num_questions, difficulty)
            questions = self.generate_statements(context, num_questions, difficulty, sentences)
            
            print("\n--- QUIZ STARTS ---\n")
            score = 0
            
            for idx, (statement, actual_label) in enumerate(questions, 1):
                print(f"Q{idx}: {statement}")
                user = self.get_user_answer()
                
                # Format input for facebook/bart-large-mnli
                input_text = f"{context} [SEP] {statement}"
                result = nli(input_text)[0]
                if result["label"] == "neutral":
                    print("Skipping ambiguous statement.\n")
                    continue
                model_label = "ENTAILMENT" if result["label"] == "entailment" else "CONTRADICTION"
                
                if model_label == "ENTAILMENT" and user == "true":
                    print("Correct!\n")
                    score += 1
                elif model_label == "CONTRADICTION" and user == "false":
                    print("Correct!\n")
                    score += 1
                else:
                    print(f"Incorrect! (Correct answer: {'True' if model_label == 'ENTAILMENT' else 'False'})\n")
            print(f"\n--- Final Score: {score}/{len(questions)} ---")

        except ValueError as e:
            print(f"Error: {e}")

def main():
    context = input(">> Enter context text: ").strip()
    try:
        num_questions = int(input(">> How many questions do you want to generate? ").strip())
    except ValueError:
        print("Please enter a valid integer for number of questions.")
        return

    difficulty = input(">> Enter difficulty level (easy / medium / hard): ").strip().lower()
    if difficulty not in ["easy", "medium", "hard"]:
        print("Invalid difficulty level.")
        return
    quiz_generator = generate_true_false()
    quiz_generator.run_quiz(context, num_questions, difficulty)

if __name__ == "__main__":
    main()
