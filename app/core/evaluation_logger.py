import csv
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2] / "data"
AUTO_LOG_FILE = BASE_DIR / "rag_automated_log.csv"
HUMAN_LOG_FILE = BASE_DIR / "human_feedback.csv"

def log_automated_metric(question, answer, time_taken, is_negative_answer, source_count):
    """Guarda métricas técnicas: Tiempo y Cobertura (Si encontró info o no)."""
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
    """Guarda la evaluación humana de calidad (1=Positivo, 0=Negativo)."""
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