# Chatbot-Grupo11

Guía rápida para arrancar la API (FastAPI + Uvicorn) y la interfaz (Streamlit) del chatbot RAG.

## Requisitos previos
- Python 3.10.x (proyecto usa Poetry).
- Dependencias instaladas: `poetry install` (HACERLO LO PRIMERO)
- Variables de entorno:
  - `UC3M_API_KEY`: clave para el LLM UC3M.
  - `API_URL`: URL base de la API para la interfaz (por defecto `http://localhost:8000`).

## 1. Levantar la API
Desde la raíz del proyecto:
```bash
poetry run uvicorn app.api.api:app --reload --host 0.0.0.0 --port 8000
```
La API quedará accesible en `http://localhost:8000`. Endpoints clave:
- `POST /chat`: recibe `{"question": "<texto>"}` y responde con `{"session_id": "...", "answer": {"role": "assistant", "content": "..."} }`.
- `GET /health`: devuelve `{ "status": "ok" }`.

## 2. Levantar la interfaz (Streamlit)
En otra terminal, también desde la raíz:
```bash
poetry run streamlit run app/interfaz/app.py
```
La interfaz abrirá en `http://localhost:8501`. Asegúrate de que en `app/interfaz/app.py` el POST apunte a `f"{API_URL}/chat"` para usar el endpoint sin prefijo.

## Variables de entorno de ejemplo
En `.env.ejemplo` tienes:
```
UC3M_API_KEY="TU API KEY"
API_URL="http://localhost:8000"
```
Crea tu `.env` con esos valores reales.

## Estructura relevante
- `app/api/`: FastAPI (CORS, rutas `/chat` y `/health`).
- `app/agente/`: lógica RAG (Chroma + LLM UC3M).
- `app/interfaz/`: interfaz Streamlit (`app.py`).
