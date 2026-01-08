import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
AUTO_FILE = DATA_DIR / "rag_automated_log.csv"
HUMAN_FILE = DATA_DIR / "human_feedback.csv"

def main():
    print("\n📊 GENERANDO INFORME DE EVALUACIÓN RAG")
    print("="*50)

    # 1. EVALUACIÓN TÉCNICA (Tiempo y Cobertura)
    if AUTO_FILE.exists():
        df_auto = pd.read_csv(AUTO_FILE)
        total_qs = len(df_auto)
        
        if total_qs > 0:
            # Tiempo
            avg_time = df_auto['time_sec'].mean()
            
            # Cobertura (Porcentaje de 'found_info' == True)
            found_info_count = df_auto[df_auto['found_info'] == True].shape[0]
            coverage = (found_info_count / total_qs) * 100
            
            print(f"✅ CRITERIO: TIEMPO DE RESPUESTA")
            print(f"   - Promedio: {avg_time:.2f} segundos")
            print(f"   - Muestras: {total_qs}")
            print("-" * 30)
            
            print(f"✅ CRITERIO: COBERTURA DOCUMENTAL")
            print(f"   - Consultas Respondidas: {found_info_count}/{total_qs}")
            print(f"   - Cobertura Total: {coverage:.1f}%")
        else:
            print("⚠️ El log automático está vacío.")
    else:
        print("❌ No se encontró el archivo de logs automáticos.")

    print("="*50)

    # 2. EVALUACIÓN DE CALIDAD (Humana)
    if HUMAN_FILE.exists():
        df_human = pd.read_csv(HUMAN_FILE)
        total_votes = len(df_human)
        
        if total_votes > 0:
            positivos = df_human[df_human['rating'] == 1].shape[0]
            quality_score = (positivos / total_votes) * 100
            
            print(f"✅ CRITERIO: CALIDAD DEL SISTEMA (Humana)")
            print(f"   - Votos Positivos (👍): {positivos}")
            print(f"   - Votos Negativos (👎): {total_votes - positivos}")
            print(f"   - Índice de Calidad: {quality_score:.1f}%")
        else:
            print("⚠️ El archivo de feedback humano está vacío.")
    else:
        print("❌ No se encontró el archivo de feedback humano.")
        
    print("="*50)

if __name__ == "__main__":
    main()