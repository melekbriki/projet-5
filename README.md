# Projet 5 : Weather Assistant avec LangGraph

## Objectif
Assistant météo qui comprend une demande en langage naturel (FR / EN / AR), trouve la ville, récupère la météo réelle et répond naturellement. Il se souvient de la ville ("Et demain ?").

## Installation
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
`.env` (le même que les projets précédents) :
```
GROQ_API_KEY=ta_cle
GROQ_MODEL=openai/gpt-oss-20b
LLM_PROVIDER=groq
```
Aucune clé météo nécessaire : l'API **Open-Meteo** est gratuite et sans clé (usage non commercial).

## Lancer
```powershell
python weather_agent.py --demo    # 6 scénarios de test avec le chemin parcouru dans le graphe
python weather_agent.py           # chat dans le terminal
python weather_agent.py --graph   # affiche le graphe (Mermaid) -> coller sur mermaid.live
streamlit run app.py              # interface web
```

## Architecture (voir `graph.png`)
- **State** (`WeatherState`) : question, intent, city, days, location, weather, answer, error.
- **Noeuds** : `understand` (LLM + sortie structurée), `geocode`, `fetch_current`, `fetch_forecast`, `respond` (LLM), `chitchat` (LLM), `ask_city`, `handle_error`.
- **Conditional edges** : après `understand` (météo ? ville connue ?), après `geocode` (erreur ? actuel ou prévision ?), après `fetch_*` (erreur ?).
- **Mémoire** : `MemorySaver` + `thread_id` -> l'état (donc la ville) persiste entre les messages.
- **Outils externes** : Open-Meteo geocoding + forecast (`requests`).

## Cas de test (`--demo`)
| Question | Chemin attendu |
|---|---|
| What's the weather in Tunis right now? | understand -> geocode -> fetch_current -> respond |
| And tomorrow? (même thread) | understand -> geocode -> fetch_forecast -> respond (ville mémorisée) |
| Quel temps fera-t-il à Paris cette semaine ? | understand -> geocode -> fetch_forecast -> respond |
| Tell me a joke | understand -> chitchat |
| Quel temps fait-il ? (sans ville) | understand -> ask_city |
| Weather in Zzqxjkv (ville inexistante) | understand -> geocode -> handle_error |

## LangChain vs LangGraph
LangChain enchaîne des étapes **linéaires** (`prompt | llm | parser`). LangGraph ajoute un **état partagé**, des **branches conditionnelles**, la gestion d'**erreurs** comme chemin du graphe, et la **persistance** (checkpointer). Ici le flux n'est pas linéaire : il dépend de l'intention, de la ville et des erreurs réseau.

## Limites et améliorations
- Une seule ville par requête, et le premier résultat du géocodage (ex. "Paris" -> Paris, France ; homonymes ambigus) -> demander confirmation ou comparer plusieurs villes.
- Prévisions limitées à 7 jours, pas de météo horaire ni historique.
- MemorySaver en RAM : perdue à la fermeture -> SqliteSaver / PostgresSaver.
- Le LLM peut mal extraire la ville (fautes, dialecte) -> few-shot examples, ou réessai.
- Pas de retry/cache sur l'API -> retry avec backoff et cache court.
- Transformer en agent à outils (ReAct) pour gérer des questions complexes : "Où aller ce week-end entre Tunis et Sousse ?".
- Human-in-the-loop (`interrupt`) pour confirmer une ville ambiguë.
