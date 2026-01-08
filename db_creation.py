import pandas as pd
import os
import shutil
import re
import ast

# Importaciones de LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# --- ⚙️ CONFIGURACIÓN ---
CARPETA_DATOS_RAW = "MovieSummaries"
CARPETA_FUENTES_TXT = "data/source_docs"
RUTA_DB_CHROMA = "data/vector_db"

# Usamos MiniLM. Al separar los documentos, este modelo es muy efectivo y rápido.
MODELO_EMBEDDINGS = "sentence-transformers/all-MiniLM-L6-v2"

# Chunk size moderado para los plots
CHUNK_SIZE = 1200  
CHUNK_OVERLAP = 200

# --- 🛠️ FUNCIONES DE LIMPIEZA Y FORMATO ---

def limpiar_nombre_archivo(texto):
    texto_seguro = re.sub(r'[\\/*?:"<>|]', "_", str(texto))
    return texto_seguro[:100]

def limpiar_fecha_completa_en(fecha_raw):
    valor = str(fecha_raw).strip()
    if not valor or valor.lower() in ['nan', 'none', 'nat']:
        return "Unknown date"
    if len(valor) >= 4 and valor[0].isdigit():
        return valor 
    return "Unknown date"

def extraer_anio(fecha_str):
    match = re.search(r'\d{4}', str(fecha_str))
    return int(match.group()) if match else 0

def parsear_generos(texto_dict):
    try:
        if pd.isna(texto_dict) or str(texto_dict) == 'nan':
            return "Unknown Genre"
        diccionario = ast.literal_eval(str(texto_dict))
        valores = list(diccionario.values())
        return ", ".join(valores) if valores else "Unknown Genre"
    except:
        return "Unknown Genre"

def formatear_dolares_limpio(valor):
    try:
        s_val = str(valor).strip()
        if not s_val or s_val.lower() in ['nan', 'none', 'unknown revenue', '0']:
            return "Unknown Box Office"
        numero = float(s_val)
        return "${:,.0f}".format(numero)
    except:
        return "Unknown Box Office"

def limpiar_edad(valor):
    try:
        s_val = str(valor).strip()
        if not s_val or s_val.lower() == 'nan':
            return "N/A"
        return str(int(float(s_val)))
    except:
        return "N/A"

def formatear_reparto(row):
    actor = str(row['actor_name']).strip()
    if not actor or actor.lower() == 'nan': 
        actor = "Unknown Actor"
    personaje = str(row['char_name']).strip()
    if not personaje or personaje.lower() == 'nan':
        personaje = "Unknown Role"
    edad = limpiar_edad(row['actor_age'])
    return f"{actor} as {personaje} (Age: {edad})"

# --- 🔄 CARGA DE DATOS ---

def cargar_y_procesar_tablas():
    print("1️⃣  LOADING DATA (Pandas)...")
    try:
        df_plots = pd.read_csv(os.path.join(CARPETA_DATOS_RAW, "plot_summaries.txt"), sep="\t", header=None, names=["movie_id", "plot"])
        
        df_meta = pd.read_csv(
            os.path.join(CARPETA_DATOS_RAW, "movie.metadata.tsv"), 
            sep="\t", header=None, 
            usecols=[0, 2, 3, 4, 5, 8], 
            names=["movie_id", "name", "release_date", "revenue", "runtime", "genres"]
        )
        
        print("   ...Reading character metadata...")
        df_chars = pd.read_csv(
            os.path.join(CARPETA_DATOS_RAW, "character.metadata.tsv"), 
            sep="\t", header=None, 
            usecols=[0, 3, 8, 9],  
            names=["movie_id", "char_name", "actor_name", "actor_age"]
        )
    except FileNotFoundError:
        print(f"❌ Error: Cannot find '{CARPETA_DATOS_RAW}'.")
        exit()

    print("   ...Processing metadata...")
    df_meta['genres'] = df_meta['genres'].apply(parsear_generos)
    df_meta['release_date'] = df_meta['release_date'].apply(limpiar_fecha_completa_en)
    df_meta['runtime'] = df_meta['runtime'].fillna("Unknown")

    print("   ...Grouping actors & characters...")
    df_chars['actor_name'] = df_chars['actor_name'].fillna("Unknown Actor")
    df_chars['char_name'] = df_chars['char_name'].fillna("Unknown Role")
    df_chars['formatted_role'] = df_chars.apply(formatear_reparto, axis=1)
    
    # Agrupar reparto
    df_cast_grouped = df_chars.groupby('movie_id')['formatted_role'].apply(lambda x: '; '.join(x)).reset_index()
    df_cast_grouped.rename(columns={'formatted_role': 'cast_full'}, inplace=True)

    # Merge final
    df_temp = pd.merge(df_plots, df_meta, on='movie_id')
    df_final = pd.merge(df_temp, df_cast_grouped, on='movie_id', how='left')
    df_final['cast_full'] = df_final['cast_full'].fillna("Cast details unavailable")
    
    return df_final

# --- 🚀 PROCESO PRINCIPAL ---

def main():
    df = cargar_y_procesar_tablas()
    
    if os.path.exists(RUTA_DB_CHROMA):
        shutil.rmtree(RUTA_DB_CHROMA)
        print(f"🗑️  Old database deleted.")
        
    if not os.path.exists(CARPETA_FUENTES_TXT):
        os.makedirs(CARPETA_FUENTES_TXT)

    print("\n2️⃣  GENERATING ENRICHED DOCUMENTS (CAST CHUNKING STRATEGY)...")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs_para_chroma = []
    
    count = 0
    for index, row in df.iterrows():
        peli_id = str(row['movie_id'])
        titulo = str(row['name'])
        fecha = str(row['release_date'])
        anio = extraer_anio(fecha)
        plot = str(row['plot'])
        generos = str(row['genres'])
        runtime = str(row['runtime']) + " min"
        revenue_raw = row['revenue']
        revenue_fmt = formatear_dolares_limpio(revenue_raw)
        reparto_completo_str = str(row['cast_full'])
        
        nombre_archivo = f"{peli_id}_{limpiar_nombre_archivo(titulo)}.txt"

        # 1. GENERAR ARCHIVO FÍSICO (Referencia humana)
        cabecera_archivo = (
            f"TITLE: {titulo}\n"
            f"RELEASE DATE: {fecha}\n"
            f"GENRES: {generos}\n"
            f"BOX OFFICE: {revenue_fmt}\n"
            f"FULL CAST: {reparto_completo_str}\n"
        )
        ruta_txt = os.path.join(CARPETA_FUENTES_TXT, nombre_archivo)
        with open(ruta_txt, "w", encoding="utf-8") as f:
            f.write(f"{cabecera_archivo}\nFULL PLOT SUMMARY:\n{plot}")

        # ---------------------------------------------------------
        # ESTRATEGIA: CAST CHUNKING (Fragmentación del Reparto)
        # ---------------------------------------------------------
        # Dividimos el reparto en grupos de 20 para evitar el truncamiento del embedding
        reparto_lista = reparto_completo_str.split("; ")
        chunk_size_cast = 20
        
        if not reparto_lista or len(reparto_lista) == 0:
            reparto_chunks = ["Unknown Cast"]
        else:
            reparto_chunks = [
                "; ".join(reparto_lista[i:i + chunk_size_cast]) 
                for i in range(0, len(reparto_lista), chunk_size_cast)
            ]

        # Crear documentos de ficha técnica por cada lote de actores
        for i, cast_batch in enumerate(reparto_chunks):
            contenido_ficha = (
                f"TITLE: {titulo}\n"
                f"RELEASE DATE: {fecha}\n"
                f"GENRES: {generos}\n"
                f"RUNTIME: {runtime}\n"
                f"BOX OFFICE: {revenue_fmt}\n"
                f"CAST PART {i+1}: {cast_batch}\n"
                f"CONTENT TYPE: Movie Technical Specs and Cast List.\n"
            )
            
            doc_ficha = Document(
                page_content=contenido_ficha,
                metadata={
                    "movie_id": peli_id,
                    "name": titulo,
                    "genres": generos,
                    "year": anio,
                    "box_office": revenue_fmt,
                    "source": nombre_archivo,
                    "doc_type": "specs"
                }
            )
            docs_para_chroma.append(doc_ficha)

        # ---------------------------------------------------------
        # ESTRATEGIA: DOCUMENTOS DE TRAMA (PLOT)
        # ---------------------------------------------------------
        cabecera_plot = (
            f"TITLE: {titulo}\n"
            f"GENRES: {generos}\n"
            f"YEAR: {anio}\n"
        )

        chunks_plot = text_splitter.split_text(plot)
        for i, chunk in enumerate(chunks_plot):
            contenido_vector = (
                f"{cabecera_plot}\n"
                f"PLOT FRAGMENT ({i+1}/{len(chunks_plot)}):\n"
                f"{chunk}"
            )
            doc_plot = Document(
                page_content=contenido_vector,
                metadata={
                    "movie_id": peli_id,
                    "name": titulo,
                    "genres": generos,
                    "year": anio,
                    "box_office": revenue_fmt,
                    "source": nombre_archivo,
                    "doc_type": "plot"
                }
            )
            docs_para_chroma.append(doc_plot)
            
        count += 1
        if count % 2000 == 0:
            print(f"   ⏳ Processed {count} movies...")

    print(f"\n3️⃣  INDEXING IN CHROMA ({len(docs_para_chroma)} docs)...")
    print(f"   Model: {MODELO_EMBEDDINGS}")
    
    embeddings = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDINGS)
    # Batch size ajustado para evitar sobrecarga de memoria
    batch_size = 5000 
    vector_db = Chroma(embedding_function=embeddings, persist_directory=RUTA_DB_CHROMA)
    
    for i in range(0, len(docs_para_chroma), batch_size):
        lote = docs_para_chroma[i : i + batch_size]
        print(f"   -> Indexing batch {i}...")
        vector_db.add_documents(lote)

    print(f"\n🎉 DONE! Database ready at '{RUTA_DB_CHROMA}'")

if __name__ == "__main__":
    main()