import json
from embeddings.vector import Vector
from summary.summarization import SummaryGenerator

def process_document(v: Vector):
    """
    Process the document available in the assets/docs directory.
    This function initializes chromadb and leverages llamindex to process the document, generate embeddings and store them in the ChromaVectorStore.
    Args:
        v (Vector): The Vector object.
    Returns:
        None
    """
    v.process_document()
    print("Document processed successfully")

def execute_query(v: Vector, question_file):
    """
    Execute queries defined in assets/questions/tasks4.json on the processed document.
    Args:
       v (Vector): The Vector object.
    Returns:
        list: A list of responses to the queries.
    """
    responses = []
    with open(question_file) as f:
        questions = json.load(f)
        for question in questions:
            response = v.query_document(question)
            responses.append({"question": question, "answer": response})
    return responses

def generate_recap(responses: list):
    """
    Generate a recap from the responses.
    Args:
        responses (list): A list of responses to the queries.
    Returns:
        str: A string containing the summary of the responses.
    """
    summary_generator = SummaryGenerator()
    return summary_generator.summarize(responses)

def main():
    v = Vector("assets/docs")
    process_document(v)
    responses = execute_query(v, "assets/questions/task4.json")
    summary = generate_recap(responses)
    print(summary)

if __name__ == "__main__":
    main()


