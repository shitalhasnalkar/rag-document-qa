
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rag_engine import answer_question 
app = FastAPI(
    title="RAG Document Q&A API",
    description="Ask questions about an Employee Handbook.",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/")
def home():
    return {
        "message": "RAG Document Q&A API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}



@app.post("/ask")
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    try:
        return answer_question(request.question)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        print(f"RAG request failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Failed to answer the question. Check the server terminal.",
        ) from exc

    return {
        "question": request.question,
        "message": "FastAPI is working. RAG integration comes next.",
    }