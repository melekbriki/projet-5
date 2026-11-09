"""Projet 5 : Weather Assistant avec LangGraph.

Graphe :
  START -> understand --(intent/city ?)--> geocode -> fetch_current / fetch_forecast -> respond -> END
           understand --> ask_city -> END        (ville manquante)
           understand --> chitchat -> END        (question hors meteo)
           geocode / fetch_* --(erreur ?)--> handle_error -> END

Donnees meteo : Open-Meteo (gratuit, sans cle API).
Memoire : MemorySaver (checkpointer) -> "Et demain ?" se souvient de la ville.
"""
import os
import sys
from typing import Literal, Optional, TypedDict

import requests
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


# ----------------------------------------------------------------- LLM
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


def structured(llm, schema):
    """Sortie structuree : json_schema (fiable avec gpt-oss sur Groq),
    avec repli automatique sur le function calling si ca echoue."""
    return llm.with_structured_output(schema, method="json_schema").with_fallbacks(
        [llm.with_structured_output(schema)]
    )


# ----------------------------------------------------------------- STATE
class WeatherState(TypedDict, total=False):
    """L'etat partage entre tous les noeuds du graphe."""

    question: str
    intent: str  # current | forecast | other
    city: Optional[str]
    days: int
    location: Optional[dict]
    weather: Optional[dict]
    answer: str
    error: Optional[str]


class Intent(BaseModel):
    intent: Literal["current", "forecast", "other"] = Field(
        description="current = weather now/today; forecast = tomorrow or coming days; "
        "other = not about weather"
    )
    city: Optional[str] = Field(
        None, description="City mentioned by the user, written in Latin letters, or null"
    )
    days: int = Field(
        1, ge=1, le=7,
        description="Number of days from today needed (tomorrow=2, this week=7, today=1)",
    )


INTENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You analyse requests sent to a weather assistant. "
                   "The user may write in English, French, Arabic or Tunisian dialect. "
                   "Extract the intent, the city and the number of days."),
        ("human", "{question}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a friendly weather assistant. Answer the question using ONLY "
                   "the weather data provided. Reply in the SAME language as the question. "
                   "Be concise (2 to 4 sentences), temperatures in °C, and add one practical "
                   "tip (umbrella, jacket, sunscreen...) when relevant."),
        ("human", "Question: {question}\n\nWeather data:\n{data}"),
    ]
)

CHITCHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a friendly weather assistant. The user said something that is "
                   "not a weather request. Reply briefly in their language and invite them "
                   "to ask about the weather in a city."),
        ("human", "{question}"),
    ]
)

# Codes meteo WMO (Open-Meteo) -> description
WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "violent showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm with hail",
}


def _describe(code) -> str:
    return WMO.get(code, f"weather code {code}")


# ----------------------------------------------------------------- OUTILS (API)
def geocode_city(city: str) -> Optional[dict]:
    r = requests.get(
        GEOCODE_URL,
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=10,
    )
    r.raise_for_status()
    results = r.json().get("results")
    if not results:
        return None
    x = results[0]
    return {
        "name": x["name"],
        "country": x.get("country", ""),
        "latitude": x["latitude"],
        "longitude": x["longitude"],
    }


def get_current(lat: float, lon: float) -> dict:
    r = requests.get(
        FORECAST_URL,
        params={
            "latitude": lat, "longitude": lon, "timezone": "auto",
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                       "wind_speed_10m,weather_code",
        },
        timeout=10,
    )
    r.raise_for_status()
    c = r.json()["current"]
    return {
        "time": c["time"],
        "temperature_c": c["temperature_2m"],
        "feels_like_c": c["apparent_temperature"],
        "humidity_percent": c["relative_humidity_2m"],
        "wind_kmh": c["wind_speed_10m"],
        "condition": _describe(c["weather_code"]),
    }


def get_forecast(lat: float, lon: float, days: int) -> list:
    r = requests.get(
        FORECAST_URL,
        params={
            "latitude": lat, "longitude": lon, "timezone": "auto",
            "forecast_days": max(1, min(days, 7)),
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                     "precipitation_probability_max",
        },
        timeout=10,
    )
    r.raise_for_status()
    d = r.json()["daily"]
    return [
        {
            "date": d["time"][i],
            "min_c": d["temperature_2m_min"][i],
            "max_c": d["temperature_2m_max"][i],
            "rain_probability_percent": d["precipitation_probability_max"][i],
            "condition": _describe(d["weather_code"][i]),
        }
        for i in range(len(d["time"]))
    ]


# ----------------------------------------------------------------- GRAPHE
def build_graph(provider: str = "groq"):
    # ---- noeuds : chacun recoit l'etat et renvoie les champs a mettre a jour
    def understand(state: WeatherState):
        llm = structured(get_llm(provider), Intent)
        res = (INTENT_PROMPT | llm).invoke({"question": state["question"]})
        return {
            "intent": res.intent,
            # si l'utilisateur ne cite pas de ville, on garde celle du tour precedent
            "city": res.city or state.get("city"),
            "days": res.days,
            # remise a zero des champs du tour precedent
            "location": None, "weather": None, "answer": "", "error": None,
        }

    def geocode(state: WeatherState):
        try:
            loc = geocode_city(state["city"])
        except requests.RequestException as e:
            return {"error": f"Service de géocodage indisponible ({e.__class__.__name__})."}
        if loc is None:
            return {"error": f"Ville introuvable : « {state['city']} »."}
        return {"location": loc}

    def fetch_current(state: WeatherState):
        loc = state["location"]
        try:
            data = get_current(loc["latitude"], loc["longitude"])
        except (requests.RequestException, KeyError) as e:
            return {"error": f"Service météo indisponible ({e.__class__.__name__})."}
        return {"weather": {"type": "current", "city": loc["name"],
                            "country": loc["country"], "data": data}}

    def fetch_forecast(state: WeatherState):
        loc = state["location"]
        try:
            data = get_forecast(loc["latitude"], loc["longitude"], state.get("days", 3))
        except (requests.RequestException, KeyError) as e:
            return {"error": f"Service météo indisponible ({e.__class__.__name__})."}
        return {"weather": {"type": "forecast", "city": loc["name"],
                            "country": loc["country"], "data": data}}

    def respond(state: WeatherState):
        msg = (ANSWER_PROMPT | get_llm(provider, 0.3)).invoke(
            {"question": state["question"], "data": state["weather"]}
        )
        return {"answer": msg.content}

    def chitchat(state: WeatherState):
        msg = (CHITCHAT_PROMPT | get_llm(provider, 0.5)).invoke(
            {"question": state["question"]}
        )
        return {"answer": msg.content}

    def ask_city(state: WeatherState):
        return {"answer": "De quelle ville parles-tu ? / Which city do you mean?"}

    def handle_error(state: WeatherState):
        return {"answer": f"Désolé, je n'ai pas pu répondre : {state['error']}"}

    # ---- aretes conditionnelles : fonctions qui renvoient le NOM du prochain noeud
    def route_after_understand(state: WeatherState) -> str:
        if state["intent"] == "other":
            return "chitchat"
        if not state.get("city"):
            return "ask_city"
        return "geocode"

    def route_after_geocode(state: WeatherState) -> str:
        if state.get("error"):
            return "handle_error"
        return "fetch_current" if state["intent"] == "current" else "fetch_forecast"

    def route_after_fetch(state: WeatherState) -> str:
        return "handle_error" if state.get("error") else "respond"

    g = StateGraph(WeatherState)
    for name, fn in [
        ("understand", understand), ("geocode", geocode),
        ("fetch_current", fetch_current), ("fetch_forecast", fetch_forecast),
        ("respond", respond), ("chitchat", chitchat),
        ("ask_city", ask_city), ("handle_error", handle_error),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "understand")
    g.add_conditional_edges(
        "understand", route_after_understand,
        {"chitchat": "chitchat", "ask_city": "ask_city", "geocode": "geocode"},
    )
    g.add_conditional_edges(
        "geocode", route_after_geocode,
        {"handle_error": "handle_error", "fetch_current": "fetch_current",
         "fetch_forecast": "fetch_forecast"},
    )
    for node in ("fetch_current", "fetch_forecast"):
        g.add_conditional_edges(
            node, route_after_fetch, {"handle_error": "handle_error", "respond": "respond"}
        )
    for node in ("respond", "chitchat", "ask_city", "handle_error"):
        g.add_edge(node, END)

    # MemorySaver : sauvegarde l'etat par thread_id (memoire de conversation)
    return g.compile(checkpointer=MemorySaver())


def ask(graph, question: str, thread_id: str = "default"):
    """Execute le graphe. Retourne (reponse, liste des noeuds traverses)."""
    config = {"configurable": {"thread_id": thread_id}}
    path = []
    for update in graph.stream({"question": question}, config, stream_mode="updates"):
        path.extend(update.keys())
    answer = graph.get_state(config).values["answer"]
    return answer, path


# ----------------------------------------------------------------- DEMO / CLI
DEMO = [
    ("t1", "What's the weather in Tunis right now?"),
    ("t1", "And tomorrow?"),                       # memoire : garde la ville
    ("t2", "Quel temps fera-t-il à Paris cette semaine ?"),
    ("t3", "Tell me a joke"),                      # hors sujet -> chitchat
    ("t4", "Quel temps fait-il ?"),                # pas de ville -> ask_city
    ("t5", "Weather in Zzqxjkv"),                  # ville inconnue -> handle_error
]


def run_demo(provider):
    graph = build_graph(provider)
    for thread, q in DEMO:
        answer, path = ask(graph, q, thread)
        print(f"\n[{thread}] User: {q}")
        print("      chemin :", " -> ".join(["START"] + path + ["END"]))
        print(f"      Bot    : {answer}")


def chat_cli(provider):
    graph = build_graph(provider)
    print("Assistant météo prêt. /exit pour quitter.\n")
    while True:
        q = input("Vous : ").strip()
        if q == "/exit":
            break
        if q:
            answer, path = ask(graph, q, "cli")
            print(f"  ({' -> '.join(path)})\nBot  : {answer}\n")


if __name__ == "__main__":
    provider = os.getenv("LLM_PROVIDER", "groq")
    if "--graph" in sys.argv:
        print(build_graph(provider).get_graph().draw_mermaid())
    elif "--demo" in sys.argv:
        run_demo(provider)
    else:
        chat_cli(provider)