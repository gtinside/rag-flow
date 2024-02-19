# RAG Flow
A rag flow that takes as a research paper and can be queried with questions defined in a given file.It includes a script that uses the answers to the questions to write a recap of the paper.

## Dependencies
- chromadb: Embeddings database
- llama-index: Data framework, that makes it easy to work with LLMs
- Research Paper (```assets/docs```)
- List of Questions (```assets/questions```)

## How to run it?
- ```pip install -r requirements.txt```
- ```python main.py```

## Workflow
1. In-memory chromadb is instantiated and a collection is created
2. Using llama-index embeddings for the given paper are created with chunk size as 100 and chunk overlap as 10
3. A script iterates through all the given questions and query chromadb collection. 
4. All question and corresponding answers are then passed to a summary generator that leverage OpenAI to generate a recap of the paper

## Sample Output
```

The research paper introduces meta-prompting as a novel approach to problem-solving, aiming to harness the collective intelligence
of expert personas for dynamic and effective solutions. Unlike standard prompting techniques, meta-prompting consistently outperforms
traditional zero-shot methods, especially in tasks requiring heuristic or trial-and-error strategies. Expert models serve as conduits for
knowledge integration and verification loops within this framework. The integration of a Python interpreter expands the versatility of meta-prompting,
broadening its potential applications to various tasks. The study highlights meta-prompting's superior performance compared to other zero-shot methods,
particularly in complex problem-solving and creative writing tasks. However, the paper also notes limitations such as cost efficiency, scalability, and
information transfer challenges that need to be addressed for optimal implementation and success.

```
