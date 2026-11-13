"""
The RTL Doctor portal: a chat on the left, a dashboard on the right.

    python -m portal.app            # then open http://127.0.0.1:8000
    python -m portal.app --port 9000

The chat model is set in `.env` (see portal/llm.py and .env.example). Without
one, a stand-in answers that can only run the tools.
"""
import argparse
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from portal import agent
from portal.llm import LLMError, get_llm

FILE_TYPES = (".v", ".sv")
MAX_FILE_BYTES = 100_000
MAX_MESSAGES = 40               # turns of conversation sent along with a question
MAX_MESSAGE_CHARS = 4000

app = FastAPI(title="RTL Doctor")


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=MAX_MESSAGE_CHARS)


class SourceFile(BaseModel):
    name: str = Field(max_length=200)
    code: str


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=MAX_MESSAGES)
    file: SourceFile | None = None


def check_file(file):
    """Input guardrail: only Verilog source files of a sensible size."""
    if not file.name.lower().endswith(FILE_TYPES):
        raise HTTPException(400, f"Only {' and '.join(FILE_TYPES)} files are accepted.")
    size = len(file.code.encode("utf-8"))
    if size > MAX_FILE_BYTES:
        raise HTTPException(400, f"The file is {size:,} bytes. The limit is {MAX_FILE_BYTES:,}.")
    if not file.code.strip() or "\x00" in file.code:
        raise HTTPException(400, "The file is empty or is not text.")


@app.get("/api/status")
def status():
    """What the page needs to know before the first question."""
    return {"model": get_llm().name, "file_types": FILE_TYPES, "max_file_bytes": MAX_FILE_BYTES}


@app.post("/api/chat")
def chat(request: ChatRequest):
    if request.messages[-1].role != "user":
        raise HTTPException(400, "The last message has to be the user's question.")
    if request.file:
        check_file(request.file)
    try:
        return agent.answer([m.model_dump() for m in request.messages],
                            request.file.model_dump() if request.file else None, get_llm())
    except LLMError as e:
        raise HTTPException(502, f"The language model did not answer: {e}.") from e


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="page")

if __name__ == "__main__":
    import uvicorn
    p = argparse.ArgumentParser(description="Run the RTL Doctor portal.")
    p.add_argument("--port", default=8000, type=int)
    uvicorn.run(app, host="127.0.0.1", port=p.parse_args().port)
