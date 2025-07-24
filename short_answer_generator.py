import torch
import random
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

class QuestionGenerator:
    def __init__(self, model_name='deepset/roberta-base-squad2'):
        """
        Initialize question generation system
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' # Detect and set device
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create QA pipeline
        self.qa_pipeline = pipeline('question-answering', model=self.model, tokenizer=self.tokenizer,device=0 if self.device == 'cuda' else -1)
        
        # Question templates
        self.question_templates = ["What is the main idea of","Who is responsible for","When did this occur","Where does this take place","Why is this important","How does this work","What are the key features of","Explain the significance of","What is the purpose of","Describe the process of"]

    def generate_questions(self, context, num_questions=3, difficulty='medium'):
        """
        Generate multiple questions from context
        """
        generated_questions = []
        attempts = 0
        max_attempts = num_questions * 10

        while len(generated_questions) < num_questions and attempts < max_attempts:
            try:
                template = random.choice(self.question_templates)    # Select random template
                words = context.split()      # Create question
                start_index = random.randint(0, max(0, len(words) - 5))
                full_question = f"{template} {' '.join(words[start_index:start_index+5])}?"
                result = self.qa_pipeline(question=full_question, context=context)   # Get answer
                
                # Validate result
                if (result['answer'] and len(result['answer']) > 3 and result['score'] > 0.5 and not any(q['answer'] == result['answer'] for q in generated_questions)):
                    generated_questions.append({'question': full_question,'answer': result['answer'],'confidence': result['score']})
                attempts += 1
            
            except Exception as e:
                print(f"Question generation error: {e}")
                attempts += 1
        return generated_questions

    def display_questions(self, questions):
        """
        Display generated questions
        """
        print("\n--- Generated Questions ---")
        for idx, q in enumerate(questions, 1):
            print(f"Q{idx}: {q['question']}")
            print(f"Expected keyword: {q['answer']} \n")

def get_user_input():
    """
    Get user input for question generation
    """
    print("\n--- Interactive Question Generator ---")
    print("\n>> Enter the context for question generation: ")  # Context input
    context = input().strip()
    
    # Number of questions
    while True:
        try:
            num_questions = int(input("\n>> How many questions do you want? (1-10): "))
            if 1 <= num_questions <= 10:
                break
            else:
                print("Please enter a number between 1 and 10.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    return context, num_questions

def main():
    # Initialize generator
    generator = QuestionGenerator()
    
    while True:
        try:
            context, num_questions = get_user_input()          # Get user input
            questions = generator.generate_questions(context, num_questions=num_questions)    # Generate questions
            if questions:                          # Display questions
                generator.display_questions(questions)
            else:
                print("Could not generate questions. Please try a different context.")
            
            # Continue option
            continue_choice = input("Generate more questions? (yes/no): ").lower()
            if continue_choice not in ['yes', 'y']:
                break
        except Exception as e:
            print(f"An error occurred: {e}")
    print("Thank you for using the Question Generator!")

if __name__ == "__main__":
    main()
import torch
import random
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

class QuestionGenerator:
    def __init__(self, model_name='distilbert-base-uncased-distilled-squad'):
        """
        Initialize question generation system using a stable QA model
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Create QA pipeline
        self.qa_pipeline = pipeline(
            'question-answering',
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == 'cuda' else -1
        )

        # Sample templates to simulate natural QA generation
        self.question_templates = [
            "What is the main idea of",
            "Who is responsible for",
            "When did this occur",
            "Where does this take place",
            "Why is this important",
            "How does this work",
            "What are the key features of",
            "Explain the significance of",
            "What is the purpose of",
            "Describe the process of"
        ]

    def generate_questions(self, context, num_questions=3, difficulty='medium'):
        """
        Generate short answer questions based on provided context
        """
        generated_questions = []
        attempts = 0
        max_attempts = num_questions * 10

        while len(generated_questions) < num_questions and attempts < max_attempts:
            try:
                template = random.choice(self.question_templates)
                words = context.split()
                start_index = random.randint(0, max(0, len(words) - 5))
                snippet = ' '.join(words[start_index:start_index + 5])
                full_question = f"{template} {snippet}?"

                result = self.qa_pipeline(question=full_question, context=context)

                # Validate and deduplicate
                if (
                    result['answer']
                    and len(result['answer']) > 3
                    and result['score'] > 0.5
                    and not any(q['answer'].lower() == result['answer'].lower() for q in generated_questions)
                ):
                    generated_questions.append({
                        'question': full_question,
                        'answer': result['answer'],
                        'confidence': result['score']
                    })
                attempts += 1

            except Exception as e:
                print(f"Question generation error: {e}")
                attempts += 1

        return generated_questions

    def display_questions(self, questions):
        print("\n--- Generated Questions ---")
        for idx, q in enumerate(questions, 1):
            print(f"Q{idx}: {q['question']}")
            print(f"Expected keyword: {q['answer']} \n")

# Run this if testing standalone
if __name__ == "__main__":
    print("\n>> Enter the context for question generation: ")
    context = input().strip()

    while True:
        try:
            num_q = int(input("\n>> How many questions do you want? (1-10): "))
            if 1 <= num_q <= 10:
                break
            print("Please enter a number between 1 and 10.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    generator = QuestionGenerator()
    questions = generator.generate_questions(context, num_questions=num_q)

    if questions:
        generator.display_questions(questions)
    else:
        print("❌ Could not generate any questions.")
