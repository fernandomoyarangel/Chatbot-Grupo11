import hashlib
import json
from pathlib import Path

def build_doc_key(metadata, content):
    """
    Construye un identificador único para un documento.

    Usa `doc_id` si está presente en los metadatos; en caso contrario,
    genera una clave basada en la fuente y un hash del contenido.

    Args:
        metadata (dict): Metadatos del documento.
        content (str): Contenido textual del documento.

    Returns:
        str: Identificador único del documento.
    """
    if metadata and metadata.get("doc_id"):
        return str(metadata["doc_id"])
    source = (metadata or {}).get("source", "unknown")
    content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
    return f"{source}:{content_hash}"

def load_topic_maps():
    """
    Carga los mapas de documentos a tópicos y la información de los tópicos.

    Lee los archivos `doc_topics.json` y `topics_info.json` si existen.

    Returns:
        tuple:
            - dict: Mapeo documento → tópico.
            - dict: Información descriptiva de los tópicos.
    """
    project_root = Path(__file__).resolve().parents[2]
    topic_dir = project_root / "app" / "topic_modeling" / "topic_model"
    doc_topics_path = topic_dir / "doc_topics.json"
    topics_info_path = topic_dir / "topics_info.json"

    doc_topics = {}
    topics_info = {}
    if doc_topics_path.exists():
        with open(doc_topics_path, "r", encoding="utf-8") as f:
            doc_topics = json.load(f)
    if topics_info_path.exists():
        with open(topics_info_path, "r", encoding="utf-8") as f:
            topics_info = json.load(f)

    return doc_topics, topics_info