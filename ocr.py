"""Projet 4 : Simple OCR avec LangChain.

Pipeline : image -> EasyOCR (texte brut) -> prompt -> LLM -> resultat
L'OCR est encapsule dans un RunnableLambda pour s'integrer dans une chaine LCEL.
"""
import argparse
import os
from functools import lru_cache
from operator import itemgetter
from typing import Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

load_dotenv()


# ---------------------------------------------------------------- LLM
def get_llm(provider: str = "groq", temperature: float = 0.0):
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            temperature=temperature,
        )
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=temperature)
    raise ValueError(f"Fournisseur inconnu : {provider}")


# ---------------------------------------------------------------- OCR
@lru_cache(maxsize=4)
def _get_reader(langs: tuple):
    """Charge le modele EasyOCR une seule fois par combinaison de langues."""
    import easyocr

    return easyocr.Reader(list(langs), gpu=False)


def extract_text(image, langs: tuple = ("en", "fr")) -> str:
    """image : chemin (str) ou contenu du fichier (bytes). Retourne le texte brut.

    Note : l'arabe ('ar') ne se combine qu'avec l'anglais ('en') dans EasyOCR.
    """
    reader = _get_reader(tuple(langs))
    lines = reader.readtext(image, detail=0, paragraph=True)
    return "\n".join(lines)


# Etape OCR sous forme de Runnable : entree {"image":..., "langs":[...]} -> texte
ocr_runnable = RunnableLambda(
    lambda x: extract_text(x["image"], tuple(x.get("langs", ("en", "fr"))))
)


# ---------------------------------------------------------------- Sortie structuree
class DocumentInfo(BaseModel):
    document_type: str = Field(description="e.g. receipt, invoice, letter, ID, other")
    vendor_or_sender: Optional[str] = Field(None, description="Company or person, if any")
    date: Optional[str] = Field(None, description="Date found in the document, if any")
    total_amount: Optional[str] = Field(None, description="Total with currency, if any")
    summary: str = Field(description="One-sentence summary of the document")
    key_points: list[str] = Field(description="3 to 5 important facts from the text")


# ---------------------------------------------------------------- Prompts
OCR_WARNING = (
    "The text below was extracted by OCR and may contain errors "
    "(wrong characters, broken lines, missing accents). "
)

PROMPTS = {
    "clean": ChatPromptTemplate.from_messages(
        [
            ("system", OCR_WARNING + "Fix the OCR errors and restore the natural "
             "layout. Do not add or invent information. Return ONLY the corrected text."),
            ("human", "{text}"),
        ]
    ),
    "summary": ChatPromptTemplate.from_messages(
        [
            ("system", OCR_WARNING + "Summarize the document in 3 short bullet points, "
             "in the same language as the document."),
            ("human", "{text}"),
        ]
    ),
    "extract": ChatPromptTemplate.from_messages(
        [
            ("system", OCR_WARNING + "Extract the requested information. "
             "Use null when a field is not present. Never invent values."),
            ("human", "{text}"),
        ]
    ),
    "ask": ChatPromptTemplate.from_messages(
        [
            ("system", OCR_WARNING + "Answer the question using ONLY the document. "
             "If the answer is not in the document, say so."),
            ("human", "Document:\n{text}\n\nQuestion: {question}"),
        ]
    ),
}


def structured(llm, schema):
    """Sortie structuree : json_schema (fiable avec gpt-oss sur Groq),
    avec repli automatique sur le function calling si ca echoue."""
    return llm.with_structured_output(schema, method="json_schema").with_fallbacks(
        [llm.with_structured_output(schema)]
    )


def build_chain(task: str = "summary", provider: str = "groq"):
    """Chaine LCEL complete : OCR -> prompt -> LLM -> parser.

    Entree : {"image": chemin|bytes, "langs": ["en","fr"], "question": "..."}
    """
    if task == "text":  # OCR seul, sans LLM
        return ocr_runnable
    if task not in PROMPTS:
        raise ValueError(f"Tache inconnue : {task}")

    llm = get_llm(provider)
    inputs = {
        "text": ocr_runnable,
        "question": lambda x: x.get("question", ""),
    }
    if task == "extract":
        return inputs | PROMPTS[task] | structured(llm, DocumentInfo)
    return inputs | PROMPTS[task] | llm | StrOutputParser()


def analyze(image, task="summary", langs=("en", "fr"), question="", provider="groq"):
    chain = build_chain(task, provider)
    return chain.invoke({"image": image, "langs": list(langs), "question": question})


# ---------------------------------------------------------------- CLI
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="OCR + LangChain")
    p.add_argument("image", help="chemin de l'image")
    p.add_argument("--task", default="summary", choices=["text", "clean", "summary", "extract", "ask"])
    p.add_argument("--lang", nargs="+", default=["en", "fr"])
    p.add_argument("--question", default="")
    a = p.parse_args()

    result = analyze(
        a.image, a.task, a.lang, a.question, os.getenv("LLM_PROVIDER", "groq")
    )
    print(result.model_dump_json(indent=2) if isinstance(result, BaseModel) else result)
