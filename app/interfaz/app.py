import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Chatbot RAG", page_icon="💬")
st.title("Chatbot RAG")

if "history" not in st.session_state:
    st.session_state.history = []

prompt = st.text_input("Pregunta", key="prompt", placeholder="Escribe tu pregunta...")

if st.button("Enviar") and prompt:
    # POST al backend
    resp = requests.post(
        f"{API_URL}/chat",  # ajusta si cambias el prefijo
        json={"question": prompt},
        timeout=30,
    )
    if resp.ok:
        data = resp.json()
        answer = data.get("answer", {}).get("content", "")
        st.session_state.history.append(("user", prompt))
        st.session_state.history.append(("assistant", answer))
    else:
        st.error(f"Error {resp.status_code}: {resp.text}")

# Mostrar historial
for role, text in st.session_state.history:
    st.markdown(f"**{role}:** {text}")
