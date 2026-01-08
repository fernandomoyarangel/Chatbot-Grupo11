import csv
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2] / "data"
AUTO_LOG_FILE = BASE_DIR / "rag_automated_log.csv"
HUMAN_LOG_FILE = BASE_DIR / "human_feedback.csv"

def log_automated_metric(question, answer, time_taken, is_negative_answer, source_count):
    """
    Registra métricas automáticas del sistema RAG.

    Guarda información sobre tiempo de respuesta y si se encontró
    información relevante en la base documental.

    Args:
        question (str): Pregunta del usuario.
        answer (str): Respuesta generada.
        time_taken (float): Tiempo de procesamiento en segundos.
        is_negative_answer (bool): Indica si no se encontró información.
        source_count (int): Número de fuentes utilizadas.
    """
    file_exists = AUTO_LOG_FILE.exists()
    
    with open(AUTO_LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "question", "time_sec", "found_info", "source_count"])
            
        writer.writerow([
            datetime.now().isoformat(),
            question,
            f"{time_taken:.2f}",
            not is_negative_answer, 
            source_count
        ])

def log_human_feedback(question, answer, rating):
    """
    Registra evaluación humana de la respuesta generada.

    Args:
        question (str): Pregunta original del usuario.
        answer (str): Respuesta generada.
        rating (int): Valoración de la respuesta (1=Positivo, 0=Negativo).
    """
    if not HUMAN_LOG_FILE.parent.exists():
        HUMAN_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    file_exists = HUMAN_LOG_FILE.exists()
    
    with open(HUMAN_LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "question", "answer", "rating"])
            
        writer.writerow([
            datetime.now().isoformat(),
            question,
            answer,
            rating
        ])