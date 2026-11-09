"""Interface Streamlit. Lancer : streamlit run app.py"""
import uuid

import streamlit as st

from weather_agent import ask, build_graph

st.set_page_config(page_title="Weather Assistant", page_icon="🌦️")
st.title("🌦️ Assistant météo (LangGraph)")

provider = st.sidebar.selectbox("Fournisseur LLM", ["groq", "gemini"])
show_path = st.sidebar.checkbox("Afficher le chemin dans le graphe", value=True)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []

if st.sidebar.button("🗑️ Nouvelle conversation"):
    st.session_state.thread_id = str(uuid.uuid4())  # nouveau thread = nouvelle memoire
    st.session_state.messages = []

# le graphe (et sa memoire MemorySaver) est garde entre les reruns Streamlit
if st.session_state.get("graph_provider") != provider:
    st.session_state.graph = build_graph(provider)
    st.session_state.graph_provider = provider

st.sidebar.markdown(
    "**Exemples**\n- Quel temps fait-il à Tunis ?\n- Et demain ?\n- Weather in Paris this week\n- شنوة الطقس في صفاقس؟"
)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if show_path and m.get("path"):
            st.caption("🧭 " + " → ".join(m["path"]))

if q := st.chat_input("Pose ta question météo..."):
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        try:
            with st.spinner("..."):
                answer, path = ask(st.session_state.graph, q, st.session_state.thread_id)
            st.markdown(answer)
            if show_path:
                st.caption("🧭 " + " → ".join(path))
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "path": path}
            )
        except Exception as e:
            st.error(f"Erreur : {e}")
