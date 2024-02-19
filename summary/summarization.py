import openai

class SummaryGenerator:
    def __init__(self):
        self.client = openai.OpenAI()

    class Summarizer:
        def summarize(self, qa_pairs):
            """
            Summarizes a research paper based on a list of question-answer pairs.

            Args:
                qa_pairs (list): A list of dictionaries containing question-answer pairs.

            Returns:
                str: The generated summary of the research paper.
            """
            q = ",".join([f"Question: {qa['question']} Answer: {qa['answer']}" for qa in qa_pairs])
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You have been provided with a list of questions and answers related to a research paper. Your task is to generate a summary of the research paper based on this information. Each question is followed by its corresponding answer"},
                    {"role": "user", "content": q}
                ])

            return completion.choices[0].message  # Return the generated summary

    # This function takes a list of question-answer pairs and generates a summary of the research paper based on the information provided.
    def summarize(self, qa_pairs):
        q = ",".join([f"Question: {qa['question']} Answer: {qa['answer']}" for qa in qa_pairs])
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You have been provided with a list of questions and answers related to a research paper. Your task is to generate a summary of the research paper based on this information. Each question is followed by its corresponding answer"},
                {"role": "user", "content": q}
            ])

        return completion.choices[0].message.content