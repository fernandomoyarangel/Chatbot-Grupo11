import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from app.core.utils import build_doc_key
import json

from app.agente.tools.rag_tools import get_vector_store

MODEL_DIR_NAME = "bertopic_model"
TOPIC_INFO_NAME = "topic_info.csv"


def ensure_chroma_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    default_vector_db = project_root / "data" / "vector_db"
    os.environ.setdefault("CHROMA_PATH", str(default_vector_db))


def get_output_dir(output_dir: Optional[Path] = None) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return output_dir or (project_root / "data" / "topic_model")


def run_topic_modeling(
    output_dir: Optional[Path] = None,
    language: str = "english",
) -> Dict[str, Any]:
    load_dotenv()
    ensure_chroma_path()

    vector_store = get_vector_store()
    data = vector_store.get(include=["documents", "metadatas", "embeddings"])
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    if len(metadatas) != len(documents):
        metadatas = (metadatas + [{}] * len(documents))[:len(documents)]

    doc_keys = [
        build_doc_key(metadatas[i], documents[i])
        for i in range(len(documents))
    ]

    if not documents:
        raise ValueError("No documents found in the vector store.")
    embeddings = data.get("embeddings")

    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    model_name = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embedding_model = SentenceTransformer(model_name)

    bertopic_language = "multilingual" if language == "spanish" else "english"
    topic_model = BERTopic(
        embedding_model=embedding_model,
        language=bertopic_language,
        verbose=True,
    )

    if embeddings is not None and len(embeddings) > 0 and len(embeddings) == len(documents):
        topics, _ = topic_model.fit_transform(documents, embeddings=embeddings)
    else:
        topics, _ = topic_model.fit_transform(documents)

    output_dir = get_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = output_dir / MODEL_DIR_NAME
    if model_dir.exists():
        shutil.rmtree(model_dir)

    topic_model.save(
        str(model_dir),
        serialization="pytorch",
        save_embedding_model=model_name,
    )

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(output_dir / TOPIC_INFO_NAME, index=False)

    doc_topics = {
        doc_keys[i]: int(topics[i])
        for i in range(len(documents))
    }
    topics_info = {}
    unique_topic_ids = sorted(set(int(t) for t in topics))
    for topic_id in unique_topic_ids:
        if topic_id == -1:
            continue
        words = topic_model.get_topic(topic_id) or []
        topics_info[str(topic_id)] = {
            "top_words": [w for w, _ in words[:10]]
        }

    with open(output_dir / "doc_topics.json", "w", encoding="utf-8") as f:
        json.dump(doc_topics, f, ensure_ascii=True, indent=2)

    with open(output_dir / "topics_info.json", "w", encoding="utf-8") as f:
        json.dump(topics_info, f, ensure_ascii=True, indent=2)
    
    return {
        "output_dir": str(output_dir),
        "model_dir": str(model_dir),
        "topic_count": int(len(topic_info)),
        "document_count": int(len(documents)),
    }


def load_topic_model(output_dir: Optional[Path] = None):
    from bertopic import BERTopic

    model_dir = get_output_dir(output_dir) / MODEL_DIR_NAME
    if not model_dir.exists():
        raise FileNotFoundError("Topic model not found. Run /topics first.")
    return BERTopic.load(str(model_dir))


def build_topic_summary(topic_model, top_n_topics: int = 10, top_n_words: int = 10) -> List[Dict[str, Any]]:
    info = topic_model.get_topic_info()
    info = info[info["Topic"] != -1]
    if top_n_topics and top_n_topics > 0:
        info = info.head(top_n_topics)

    topics = []
    for _, row in info.iterrows():
        topic_id = int(row["Topic"])
        words = topic_model.get_topic(topic_id) or []
        topics.append({
            "topic_id": topic_id,
            "count": int(row.get("Count", 0)),
            "name": str(row.get("Name", "")),
            "top_words": [word for word, _ in words[:max(top_n_words, 0)]],
        })
    return topics


def build_topic_visualization(
    top_n_topics: int = 10,
    top_n_words: int = 10,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    topic_model = load_topic_model(output_dir)
    fig = topic_model.visualize_topics()
    plot_html = fig.to_html(include_plotlyjs="inline", full_html=False)
    topics = build_topic_summary(topic_model, top_n_topics=top_n_topics, top_n_words=top_n_words)
    return {"plot_html": plot_html, "topics": topics}
