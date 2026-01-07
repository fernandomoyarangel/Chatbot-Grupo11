import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
from dotenv import load_dotenv
from app.agente.tools.rag_tools import get_vector_store
import torch
import numpy as np # Necesario para numpy load

# Nombres de archivos y directorios constantes
MODEL_DIR_NAME = "bertopic_model"
TOPIC_INFO_NAME = "topic_info.csv"
VIS_FILENAME = "topic_visualization.html"
SUMMARY_FILENAME = "topic_summary_data.json"
DOC_TOPICS_FILENAME = "doc_topics.json"

def ensure_chroma_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    default_vector_db = project_root / "data" / "vector_db"
    os.environ.setdefault("CHROMA_PATH", str(default_vector_db))

def get_output_dir(output_dir: Optional[Path] = None) -> Path:
    if output_dir:
        return output_dir
    current_dir = Path(__file__).resolve().parent
    return current_dir / "topic_model"

def run_topic_modeling(
    output_dir: Optional[Path] = None,
    language: str = "english",
) -> Dict[str, Any]:
    load_dotenv()
    ensure_chroma_path()

    # 1. Definir rutas de salida
    out_path = get_output_dir(output_dir)
    model_dir = out_path / MODEL_DIR_NAME
    

    if model_dir.exists():
        print(f"--> Modelo encontrado en: {model_dir}")
        try:
            from bertopic import BERTopic
            

            doc_count = 0
            doc_topics_path = out_path / DOC_TOPICS_FILENAME
            if doc_topics_path.exists():
                with open(doc_topics_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    doc_count = len(data)
            

            import pandas as pd
            topic_info_path = out_path / TOPIC_INFO_NAME
            topic_count = 0
            if topic_info_path.exists():
                df = pd.read_csv(topic_info_path)
                topic_count = len(df)


            if (out_path / VIS_FILENAME).exists() and (out_path / SUMMARY_FILENAME).exists():
                print("--> Todos los archivos existen. Saltando entrenamiento.")
                return {
                    "output_dir": str(out_path),
                    "model_dir": str(model_dir),
                    "topic_count": topic_count,
                    "document_count": doc_count,
                }
            else:
                print("--> El modelo existe pero faltan visualizaciones. Regenerando sin recalcular embeddings...")
                topic_model = BERTopic.load(str(model_dir))
                topic_info = topic_model.get_topic_info()
                topic_info.to_csv(out_path / TOPIC_INFO_NAME, index=False)
                topic_count = int(len(topic_info))

                fig = topic_model.visualize_topics()
                plot_html = fig.to_html(include_plotlyjs="inline", full_html=False)
                with open(out_path / VIS_FILENAME, "w", encoding="utf-8") as f:
                    f.write(plot_html)

                topics_summary = build_topic_summary(topic_model, top_n_topics=20, top_n_words=10)
                with open(out_path / SUMMARY_FILENAME, "w", encoding="utf-8") as f:
                    json.dump(topics_summary, f, ensure_ascii=True, indent=2)

                return {
                    "output_dir": str(out_path),
                    "model_dir": str(model_dir),
                    "topic_count": topic_count,
                    "document_count": doc_count,
                }

        except Exception as e:
            print(f"--> Error al verificar modelo existente ({e}). Se procederá a re-entrenar.")

    

    print("--> Iniciando proceso de carga y entrenamiento...")
    out_path.mkdir(parents=True, exist_ok=True)
    embeddings_file = out_path / "embeddings.npy"

    vector_store = get_vector_store()
    data = vector_store.get(include=["documents", "metadatas"]) # No pedimos embeddings a Chroma
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    
    if not documents:
        raise ValueError("No documents found in the vector store.")
    
    if len(metadatas) != len(documents):
        metadatas = (metadatas + [{}] * len(documents))[:len(documents)]

    from app.core.utils import build_doc_key
    doc_keys = [
        build_doc_key(metadatas[i], documents[i])
        for i in range(len(documents))
    ]


    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from bertopic.representation import KeyBERTInspired
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    spanish_stopwords = nltk_stopwords.words('spanish')
    english_stopwords = nltk_stopwords.words('english')
    
    my_custom_stopwords = [
        "cast", "date", "plot", "release", "title", "fragment", 
        "film", "movie", "cinema", "director", "directed", "starring",
        "story", "production", "known", "best", "series", "version",
        "one", "two", "three", "time", "life", "family", "man", "woman",
        "love", "world", "day", "year", "years", "new", "el", "la", "en"
    ]
    
    all_stop_words = list(set(spanish_stopwords + english_stopwords + my_custom_stopwords))
    vectorizer_model = CountVectorizer(stop_words=all_stop_words, min_df=5)

    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    representation_model = KeyBERTInspired()


    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--> Cargando modelo de embeddings en: {device.upper()}")
    
    embedding_model = SentenceTransformer(model_name, device=device)


    embeddings = None
    if embeddings_file.exists():
        print(f"--> Buscando cache de embeddings en {embeddings_file}...")
        try:
            loaded_embeddings = np.load(embeddings_file)
            if len(loaded_embeddings) == len(documents):
                embeddings = loaded_embeddings
                print("--> Embeddings cargados correctamente (CACHE HIT).")
            else:
                print("--> El número de documentos cambió. Recalculando embeddings...")
        except Exception as e:
            print(f"--> Error leyendo cache: {e}. Recalculando...")

    if embeddings is None:
        print("--> Calculando embeddings (esto puede tardar)...")
        embeddings = embedding_model.encode(documents, show_progress_bar=True)
        np.save(embeddings_file, embeddings)
        print(f"--> Embeddings guardados en {embeddings_file}")


    if model_dir.exists():
        print("--> Cargando modelo existente para regenerar visuales o finalizar...")
        topic_model = BERTopic.load(str(model_dir))
    else:
        # Instanciar y Entrenar nuevo
        print("--> Entrenando nuevo modelo BERTopic...")
        topic_model = BERTopic(
            embedding_model=embedding_model,
            language="multilingual",
            verbose=True,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=representation_model,
            nr_topics=60,
            min_topic_size=30
        )
        topics, _ = topic_model.fit_transform(documents, embeddings=embeddings)
        

        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        topic_model.save(
            str(model_dir),
            serialization="pytorch",
            save_embedding_model=model_name,
        )

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(out_path / TOPIC_INFO_NAME, index=False)


    if 'topics' not in locals():
        print("--> Infiriendo tópicos para documentos (usando embeddings en cache)...")
        topics, _ = topic_model.transform(documents, embeddings=embeddings)

    doc_topics = {
        doc_keys[i]: int(topics[i])
        for i in range(len(documents))
    }
    with open(out_path / DOC_TOPICS_FILENAME, "w", encoding="utf-8") as f:
        json.dump(doc_topics, f, ensure_ascii=True, indent=2)


    print("--> Generando visualizaciones...")
    fig = topic_model.visualize_topics()
    plot_html = fig.to_html(include_plotlyjs="inline", full_html=False)
    
    with open(out_path / VIS_FILENAME, "w", encoding="utf-8") as f:
        f.write(plot_html)

    topics_summary = build_topic_summary(topic_model, top_n_topics=20, top_n_words=10)
    with open(out_path / SUMMARY_FILENAME, "w", encoding="utf-8") as f:
        json.dump(topics_summary, f, ensure_ascii=True, indent=2)
    
    return {
        "output_dir": str(out_path),
        "model_dir": str(model_dir),
        "topic_count": int(len(topic_info)),
        "document_count": int(len(documents)),
    }

# Funciones auxiliares
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

    topics_list = []
    for _, row in info.iterrows():
        topic_id = int(row["Topic"])
        words = topic_model.get_topic(topic_id) or []
        topics_list.append({
            "topic_id": topic_id,
            "count": int(row.get("Count", 0)),
            "name": str(row.get("Name", "")),
            "top_words": [word for word, _ in words[:max(top_n_words, 0)]],
        })
    return topics_list

def build_topic_visualization(
    top_n_topics: int = 10,
    top_n_words: int = 10,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    out_path = get_output_dir(output_dir)
    html_path = out_path / VIS_FILENAME
    json_path = out_path / SUMMARY_FILENAME

    if html_path.exists() and json_path.exists():
        try:
            with open(html_path, "r", encoding="utf-8") as f:
                plot_html = f.read()
            with open(json_path, "r", encoding="utf-8") as f:
                topics = json.load(f)
            return {"plot_html": plot_html, "topics": topics}
        except Exception as e:
            print(f"Error loading cached topics: {e}, regenerating...")

    topic_model = load_topic_model(output_dir)
    fig = topic_model.visualize_topics()
    plot_html = fig.to_html(include_plotlyjs="inline", full_html=False)
    topics = build_topic_summary(topic_model, top_n_topics=top_n_topics, top_n_words=top_n_words)
    
    return {"plot_html": plot_html, "topics": topics}
