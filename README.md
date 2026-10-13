# RAG Document Q&A Application

A Retrieval-Augmented Generation (RAG) application that allows users to ask natural-language questions about documents and receive answers grounded in retrieved passages.

Built using **Python, FAISS, Sentence Transformers, FastAPI, Streamlit, and OpenAI API**.

## Overview

Traditional language models can generate answers that are not supported by your documents. This project combines semantic search with large language model (LLM) generation to provide context-aware answers based on relevant document passages.

The application retrieves relevant text chunks from a FAISS vector index, supplies them as context to an LLM, and returns an answer along with source passages.

## Features

* **Document retrieval:** Search relevant passages using semantic similarity.
* **Vector search:** Store and retrieve document embeddings using FAISS.
* **Natural-language answers:** Generate context-grounded answers using an LLM.
* **Source passages:** Display retrieved passages to help users verify answers.
* **REST API:** Expose question-answering functionality through FastAPI.
* **Interactive interface:** Use Streamlit to ask questions and view answers.
* **Reusable components:** Keep retrieval and answer generation in a dedicated RAG engine.

## Technology Stack

| Technology            | Purpose                  |
| --------------------- | ------------------------ |
| Python                | Application development  |
| Sentence Transformers | Text embeddings          |
| FAISS                 | Vector similarity search |
| OpenAI API            | Answer generation        |
| FastAPI               | REST API                 |
| Streamlit             | User interface           |
| Pickle                | Store document chunks    |
| Uvicorn               | Run the API server       |
| Git and GitHub        | Version control          |

## Architecture

```text
                 Document
                    |
                    v
                 ingest.py
                    |
                    v
          Text chunks and embeddings
                    |
                    v
           FAISS vector index
          document.index
                    |
                    v
              User question
                    |
                    v
           Sentence Transformer
                    |
                    v
             FAISS retrieval
                    |
                    v
          Relevant source passages
                    |
                    v
              LLM generation
                    |
                    v
          Answer + source passages
                    |
             +------+------+
             |             |
             v             v
          Streamlit     FastAPI
          Interface     REST API
```

## Project Structure

```text
rag-document-qa/
├── app.py
├── api.py
├── rag_engine.py
├── ingest.py
├── employee_handbook_rag_sample.txt
├── .gitignore
└── README.md
```

Generated files:

* `document.index` — FAISS vector index.
* `chunks.pkl` — Stored document chunks.

These generated files are excluded from Git by `.gitignore` and should be regenerated using the ingestion process when needed.

## Getting Started

### Prerequisites

* Windows
* Python
* Git
* An OpenAI API key with available API credits for the current implementation

### 1. Clone the repository

```bash
git clone https://github.com/shitalhasnalkar/rag-document-qa.git
cd rag-document-qa
```

### 2. Install dependencies

Install the Python packages used by the application:

```bash
python -m pip install streamlit fastapi uvicorn sentence-transformers faiss-cpu openai python-dotenv
```

If a package installation fails, check its compatibility with your Python version.

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
```

Replace the placeholder with your own API key. Never commit `.env` to GitHub or share your secret key publicly.

### 4. Generate the vector index

Run the ingestion script to process the sample document and create the files required for retrieval:

```bash
python ingest.py
```

Ensure that `document.index` and `chunks.pkl` are generated successfully before starting the application.

### 5. Run the Streamlit application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

### 6. Run the FastAPI server

Open a separate terminal in the project directory:

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

The API will be available at:

* API root: http://127.0.0.1:8000/
* Health check: http://127.0.0.1:8000/health
* Interactive API documentation: http://127.0.0.1:8000/docs

## API Usage

### Ask a question

**Endpoint:** `POST /ask`

Example request:

```json
{
  "question": "What is the leave policy?"
}
```

Example response structure:

```json
{
  "question": "What is the leave policy?",
  "answer": "The answer is generated using relevant retrieved passages.",
  "sources": []
}
```

The actual answer and source passages depend on the indexed document. The response above illustrates the structure, not a verified policy answer.

## Example Questions

After indexing the employee handbook, try questions such as:

* What is the leave policy?
* What are the working hours?
* What is the work-from-home policy?
* What are the rules for employee benefits?

The answers depend on the contents of the source document.

## Current Limitations

* Answers depend on the quality and coverage of the indexed documents.
* Missing or irrelevant source passages can reduce answer quality.
* OpenAI answer generation requires API access and available credits.
* The generated vector index and document chunks need to be available locally when running the application.

## Future Improvements

* Replace cloud-based generation with a local Ollama model.
* Support PDF and multiple-document ingestion.
* Improve chunking and retrieval quality.
* Add automated tests for retrieval and answer generation.
* Add Docker-based deployment.
* Deploy the application to a cloud platform.

## Author

**Shital Hasnalkar**

GitHub: [@shitalhasnalkar](https://github.com/shitalhasnalkar)

---

*This project was developed to explore semantic search, vector databases, retrieval-augmented generation, and API-based AI applications.*
