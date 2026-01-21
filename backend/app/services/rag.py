import os
import time
from pathlib import Path
from typing import List
from google import genai
from google.genai.errors import ServerError

PROJECT_ROOT=Path(__file__).resolve().parents[3]

from backend.app.services.retrieval import Retrievar

class RAGService:
    def __init__(
            self,
            retriever: Retrievar,
            prompt_path: Path,
            model:str="gemini-2.5-flash"
    ):
        self.retriever = retriever
        self.prompt_template=prompt_path.read_text(encoding="utf-8")
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model_name=model

    def _build_prompt(self, context: str, question: str) -> str:
        return f"""
    You are a question-answering system.
    Use ONLY the information provided in the context.
    If the answer is not present, say "I don't know based on the provided documents."

    Context:
    {context}

    Question:
    {question}

    Answer (with citations):
    """

    def _build_context(self,res:List[dict])->str:
        context=[]
        for r in res:
            block=(
                f"[Source: {r['doc_id']} | page {r['page']}]\n"
                f"{r.get('text', '')}"
            )
            context.append(block)

        return "\n\n".join(context)

    def answer(self, question: str, top_k: int = 5) -> str:
        results = self.retriever.search(question, top_k=top_k)

        if not results:
            return "No relevant information found in the documents."

        context = self._build_context(results)
        prompt = self._build_prompt(context, question)

        print("----- CONTEXT SENT TO LLM -----")
        print(context[:1500])
        print("----- END CONTEXT -----")

        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={
                        "temperature": 0.1,
                        "max_output_tokens": 1024
                    }
                )
                return response.text

            except ServerError as e:
                print(f"Gemini overloaded (attempt {attempt + 1}/3)")
                time.sleep(2)

        return (
            "The language model is temporarily overloaded. "
            "Please try again in a few moments."
        )