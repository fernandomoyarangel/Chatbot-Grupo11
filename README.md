# Chatbot-Grupo11

Guia rapida para clonar el repo y ejecutar la API (FastAPI + Uvicorn) y la interfaz (Streamlit) del chatbot.

## Requisitos
- Python 3.10.x
- Poetry instalado
- Variables de entorno:
  - UC3M_API_KEY: clave para el LLM UC3M
  - API_URL: URL base de la API (por defecto http://localhost:8000)

## 1) Descargar el repositorio
```bash
git clone <URL_DEL_REPO>
cd Chatbot-Grupo11
```

## 2) Instalar dependencias
```bash
poetry install
```

## 3) Configurar variables de entorno
Copia el archivo de ejemplo y edita los valores:
```bash
copy .env.ejemplo .env
```
En `.env` define:
```
UC3M_API_KEY="TU_API_KEY"
API_URL="http://localhost:8000"
```

## 4) Levantar la API (FastAPI)
En una terminal desde la raiz del proyecto:
```bash
poetry run uvicorn app.api.api:app --reload --host 0.0.0.0 --port 8000
```
La API queda en `http://localhost:8000`.

### Endpoints de la API
- `POST /chat`
  - Request: `{"question": "...", "language": "english"}` (language opcional: english|spanish).
  - Response: `{"session_id": "...", "answer": {"role": "assistant", "content": "...", "sources": [...], "suggestions": [...]}}`.
- `POST /topics`
  - Request: `{"language": "english"}`.
  - Response: `{"output_dir": "...", "model_dir": "...", "topic_count": 0, "document_count": 0}`.
- `POST /topics/visualize`
  - Request: `{"top_n_topics": 10, "top_n_words": 10, "rebuild_if_missing": false, "language": "english"}`.
  - Response: `{"plot_html": "<html>...", "topics": [...]}`.
- `POST /summary`
  - Request: `{"filename": "archivo.txt", "language": "english"}`.
  - Response: `{"summary": "..."}`.
- `POST /surprise`
  - Request: `{"language": "english"}`.
  - Response: `{"curiosity": "..."}`.
- `GET /health`
  - Response: `{ "status": "ok" }`.

## 5) Levantar la interfaz (Streamlit)
En otra terminal desde la raiz:
```bash
poetry run streamlit run app/interfaz/app.py
```
La interfaz abre en `http://localhost:8501`.

## 6) Verificacion rapida
- API: abre `http://localhost:8000/health`.
- Interfaz: abre `http://localhost:8501` y envia una pregunta.

## Estructura del proyecto
- `app/api/`: API FastAPI.
- `app/agente/`: logica RAG (Chroma + LLM UC3M).
- `app/topic_modeling/`: modelado y visualizacion de topicos.
- `app/interfaz/`: interfaz Streamlit.
- `data/`: datos y base vectorial.