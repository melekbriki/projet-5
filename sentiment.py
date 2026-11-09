"""Projet 2 : Sentiment Analysis avec LangChain.

Chaine : prompt | llm.with_structured_output(SentimentResult)
Le LLM renvoie directement un objet Pydantic valide (pas de parsing manuel).
"""
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()


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


# 1) Schema de sortie : le LLM DOIT remplir ces champs
class SentimentResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Overall sentiment of the text"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence between 0 and 1")
    reason: str = Field(description="One short sentence explaining the decision")


# 2) Prompt
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a sentiment analysis expert. Classify the sentiment of the user's "
            "text (reviews, opinions, feedback) as positive, negative or neutral. "
            "The text can be in any language (English, French, Arabic...). "
            "Mixed opinions: choose the dominant sentiment, or neutral if balanced. "
            "Write the reason in English.",
        ),
        ("human", "{text}"),
    ]
)


def structured(llm, schema):
    """Sortie structuree : json_schema (fiable avec gpt-oss sur Groq),
    avec repli automatique sur le function calling si ca echoue."""
    return llm.with_structured_output(schema, method="json_schema").with_fallbacks(
        [llm.with_structured_output(schema)]
    )


def build_analyzer(provider: str = "groq"):
    """Chaine LCEL : prompt -> LLM contraint au schema SentimentResult."""
    return PROMPT | structured(get_llm(provider), SentimentResult)


def analyze(text: str, provider: str = "groq") -> SentimentResult:
    return build_analyzer(provider).invoke({"text": text})


def analyze_many(texts: list[str], provider: str = "groq") -> list[SentimentResult]:
    """Traite plusieurs textes avec .batch() (execution en parallele)."""
    return build_analyzer(provider).batch(
        [{"text": t} for t in texts], config={"max_concurrency": 2}
    )


# Jeu de test etiquete pour evaluer l'application (accuracy)
TEST_CASES = [
    ("I absolutely love this phone, the battery lasts for days!", "positive"),
    ("Terrible service. The waiter was rude and the food was cold.", "negative"),
    ("The package arrived on Tuesday.", "neutral"),
    ("Le film était magnifique, j'ai adoré chaque minute.", "positive"),
    ("Produit de mauvaise qualité, il est tombé en panne après deux jours.", "negative"),
    ("الخدمة كانت ممتازة وسريعة جدا", "positive"),
    ("Great screen but the battery is awful and it overheats.", "negative"),
    ("Oh great, another delay. Just what I needed today.", "negative"),  # sarcasme
    ("The meeting is scheduled for 10 AM in room B.", "neutral"),
    ("Not bad at all, I would buy it again.", "positive"),  # negation
]


def run_tests(provider: str = "groq") -> float:
    results = analyze_many([t for t, _ in TEST_CASES], provider)
    correct = 0
    for (text, expected), res in zip(TEST_CASES, results):
        ok = res.sentiment == expected
        correct += ok
        mark = "OK " if ok else "XX "
        print(f"{mark}[{expected:8} -> {res.sentiment:8} ({res.confidence:.2f})] {text}")
        print(f"      raison : {res.reason}")
    acc = correct / len(TEST_CASES)
    print(f"\nAccuracy : {correct}/{len(TEST_CASES)} = {acc:.0%}")
    return acc


if __name__ == "__main__":
    run_tests(os.getenv("LLM_PROVIDER", "groq"))
