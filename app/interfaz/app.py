import os
import requests
import streamlit as st

# --- Configuración de la página ---
st.set_page_config(
    page_title="CineBot Expert 🎬",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Estilos CSS Personalizados (Tema Otoño/Cine) ---
st.markdown("""
<style>
    /* Fondo y tipografía general */
    .stApp {
        background-color: #FAF3E0; /* Beige suave */
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Encabezado principal */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        color: #5D4037; /* Marrón oscuro */
        margin-bottom: 0.5rem;
        text-align: center;
        text-shadow: 1px 1px 2px #d7ccc8;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #A1887F; /* Marrón más claro */
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }

    /* Ajustes adicionales para chat */
    .stChatMessage {
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        color: #000000; /* Texto negro para mejor legibilidad */
    }
    
    /* Asegurar que el texto dentro de los mensajes sea negro */
    .stChatMessage p, .stChatMessage div, .stChatMessage span {
        color: #000000 !important;
    }
    
    /* Mensaje del Usuario */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #FFF3E0; /* Naranja muy suave */
        border-left: 5px solid #E67E22; /* Borde naranja otoño */
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #4E342E; /* Marrón café intenso */
        color: #EFEBE9;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #FFCC80; /* Naranja/Dorado suave */
    }
    [data-testid="stSidebar"] p {
        color: #D7CCC8;
    }
    
    /* Input de chat */
    .stChatInputContainer {
        padding-bottom: 20px;
    }

    /* Modal de topicos */
    [data-testid="stDialog"] > div,
    [data-testid="stModal"] > div {
        width: min(95vw, 1400px) !important;
        margin: auto !important;
    }

    [data-testid="stDialog"] > div > div,
    [data-testid="stModal"] > div > div {
        max-height: 92vh !important;
        overflow: auto !important;
    }
    div[role="dialog"]{
    width: min(98vw, 1600px) !important;
    max-width: 98vw !important;
    margin: auto !important;
    }

    /* Quita padding lateral interno para ganar espacio real */
    div[role="dialog"] > div{
    padding-left: 0.75rem !important;
    padding-right: 0.75rem !important;
    }

    /* El iframe y su contenido que ocupen todo */
    iframe {
    width: 100% !important;
    min-width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Variables de Entorno ---
API_URL = os.getenv("API_URL", "http://localhost:8000")

# --- Gestión del Estado (Session State) ---
if "history" not in st.session_state:
    st.session_state.history = []

if "language" not in st.session_state:
    st.session_state.language = "English"

if "topic_plot_html" not in st.session_state:
    st.session_state.topic_plot_html = None
if "topic_topics" not in st.session_state:
    st.session_state.topic_topics = None
if "topic_error" not in st.session_state:
    st.session_state.topic_error = None
if "show_topics_modal" not in st.session_state:
    st.session_state.show_topics_modal = False

if "sources_history" not in st.session_state:
    st.session_state.sources_history = []

def clear_history():
    st.session_state.history = []
    st.session_state.sources_history = []
# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/dusk/200/clapperboard.png", output_format="PNG") 
    st.title("Taquilla & Ajustes")
    st.markdown("---")
    st.write("Gestiona tu sesión de cine.")
    
    if st.button("🗑️ Corten! (Borrar Historial)", use_container_width=True):
        clear_history()
        st.rerun()
        
    st.markdown("---")
    
    # Language selector
    st.write("🌐 **Idioma / Language**")
    language = st.selectbox(
        "Selecciona el idioma de conversación:",
        ["English", "Español"],
        index=0,
        label_visibility="collapsed"
    )
    st.session_state.language = language
    
    st.markdown("---")
    st.write("Analitica")
    if st.button("Generar topicos (BERTopic)", use_container_width=True):
        with st.spinner("Modelando topicos..."):
            try:
                lang_map = {"English": "english", "Español": "spanish"}
                resp = requests.post(
                    f"{API_URL}/topics",
                    json={"language": lang_map.get(st.session_state.language, "english")},
                    timeout=300,
                )
                if resp.ok:
                    result = resp.json()
                    st.success(f"Modelo guardado en: {result['output_dir']}")
                    st.caption(
                        f"Temas: {result['topic_count']} | Documentos: {result['document_count']}"
                    )
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except requests.exceptions.RequestException as exc:
                st.error(f"Error al llamar la API: {exc}")
    if st.button("Visualizar topicos", use_container_width=True):
        st.session_state.show_topics_modal = True
        st.session_state.topic_error = None
        st.session_state.topic_plot_html = None
        st.session_state.topic_topics = None
        with st.spinner("Generando grafico de topicos..."):
            try:
                lang_map = {"English": "english", "EspaÇñol": "spanish"}
                resp = requests.post(
                    f"{API_URL}/topics/visualize",
                    json={
                        "top_n_topics": 10,
                        "top_n_words": 10,
                        "rebuild_if_missing": False,
                        "language": lang_map.get(st.session_state.language, "english"),
                    },
                    timeout=300,
                )
                if resp.ok:
                    result = resp.json()
                    st.session_state.topic_plot_html = result.get("plot_html")
                    st.session_state.topic_topics = result.get("topics")
                    st.session_state.topic_error = None
                else:
                    st.session_state.topic_error = f"Error {resp.status_code}: {resp.text}"
            except requests.exceptions.RequestException as exc:
                st.session_state.topic_error = f"Error al llamar la API: {exc}"
    st.markdown("---")
    st.info(
        "**Info:** Soy tu guionista experto en cine. "
        "Pregúntame sobre directores, tramas, o recomendaciones utilizando RAG."
    )
    st.markdown("© 2025 CineBot Producciones")

# --- Interfaz Principal ---
st.markdown('<div class="main-header">🎬 CineBot Expert</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">"Le haré una oferta que no podrá rechazar... responder sus dudas de cine."</div>', unsafe_allow_html=True)

def render_topics_modal():
    if st.session_state.topic_error:
        st.error(st.session_state.topic_error)
    if not st.session_state.topic_plot_html and not st.session_state.topic_topics:
        st.info("No hay datos para mostrar. Genera topicos primero.")
    else:
        tabs = st.tabs(["Mapa de topicos", "Palabras clave"])
        with tabs[0]:
            if st.session_state.topic_plot_html:
                html = st.session_state.topic_plot_html

                # Forzar Plotly responsive dentro del modal
                html = f"""
                <div style="width:100%; height:100%; margin:0; padding:0;">
                <style>
                    html, body {{ margin:0; padding:0; width:100%; height:100%; overflow:hidden; }}
                    .plotly-graph-div {{ width:100% !important; height:100% !important; }}
                </style>
                {html}
                <script>
                    // Resize al cargar y cuando cambie el tamaño del modal
                    const resizePlot = () => {{
                    const gd = document.querySelector('.plotly-graph-div');
                    if (gd && window.Plotly) {{
                        window.Plotly.Plots.resize(gd);
                    }}
                    }};
                    window.addEventListener('load', resizePlot);
                    window.addEventListener('resize', resizePlot);
                    setTimeout(resizePlot, 300);
                    setTimeout(resizePlot, 800);
                </script>
                </div>
                """

                st.components.v1.html(html, height=900, scrolling=False)
                st.caption("Usa zoom y hover para explorar distancias entre topicos.")
            else:
                st.info("No hay grafico disponible.")
        with tabs[1]:
            if st.session_state.topic_topics:
                st.dataframe(
                    st.session_state.topic_topics,
                    use_container_width=True,
                    height=420,
                )
            else:
                st.info("No hay resumen de palabras.")
    if st.button("Cerrar", key="close_topics_modal"):
        st.session_state.show_topics_modal = False
        st.rerun()

if st.session_state.show_topics_modal:
    if hasattr(st, "dialog"):
        @st.dialog("Topicos")
        def _topics_dialog():
            render_topics_modal()
        _topics_dialog()
    elif hasattr(st, "modal"):
        with st.modal("Topicos"):
            render_topics_modal()
    else:
        st.warning("Tu version de Streamlit no soporta modales.")
        render_topics_modal()

# Contenedor para el historial de chat
chat_container = st.container()

def filter_sources_by_answer(answer_text, sources):
    answer_lower = answer_text.lower()
    filtered = []
    for s in sources:
        src = (s.get("source") or "").lower()
        if src and src in answer_lower:
            filtered.append(s)
    return filtered

def dedupe_sources(sources):
    seen = set()
    result = []
    for s in sources:
        key = s.get("source")
        if key and key not in seen:
            seen.add(key)
            result.append(s)
    return result

with chat_container:
    if not st.session_state.history:
        st.markdown(
            """
            <div style="text-align: center; margin-top: 50px; color: #8D6E63;">
                <h3>🍿 ¡Silencio en el set!</h3>
                <p>¿Qué película o dato cinematográfico buscamos hoy?</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    assistant_idx = 0
    for role, text in st.session_state.history:
        # Personalizar iconos según rol
        avatar = "👤" if role == "user" else "🎥"
        with st.chat_message(role, avatar=avatar):
            st.markdown(text)
            if role == "assistant":
                sources = []
                if assistant_idx < len(st.session_state.sources_history):
                    sources = st.session_state.sources_history[assistant_idx]
                assistant_idx += 1

                if sources:
                    with st.expander("Sources and topics"):
                        for s in sources:
                            st.markdown(f"Source: {s.get('source', 'Unknown')}")
                            st.markdown(f"Topic: {s.get('topic_id', 'N/A')}")
                            words = ", ".join(s.get("topic_words", []))
                            if words:
                                st.markdown(f"Keywords: {words}")
                            st.markdown("---")

# Input de chat (siempre abajo)
# Dynamic placeholder based on language
placeholder_text = "Write your script here..." if st.session_state.language == "English" else "Escribe tu guion aquí..."

if prompt := st.chat_input(placeholder_text):
    # Agregar mensaje del usuario al historial
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Respuesta del asistente
    with st.chat_message("assistant", avatar="🎥"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Consultando la filmoteca..." if st.session_state.language == "Español" else "Consulting the film library..."):
            sources = []
            try:
                # Map language to API format
                lang_map = {"English": "english", "Español": "spanish"}
                
                resp = requests.post(
                    f"{API_URL}/chat",
                    json={"question": prompt, "language": lang_map[st.session_state.language]},
                    timeout=30,
                )
                if resp.ok:
                    data = resp.json()
                    full_response = data.get("answer", {}).get("content", "")
                    sources = data.get("answer", {}).get("sources", [])
                else:
                    error_msg = "Corte! Error" if st.session_state.language == "Español" else "Cut! Error"
                    problem_msg = "El proyeccionista tuvo un problema." if st.session_state.language == "Español" else "The projectionist had a problem."
                    full_response = f"⚠️ **{error_msg} {resp.status_code}:** {problem_msg}\n\n`{resp.text}`"
            except requests.exceptions.RequestException as e:
                error_msg = "Error de producción:" if st.session_state.language == "Español" else "Production error:"
                full_response = f"🚨 **{error_msg}** {e}"

        # Mostrar respuesta final
        message_placeholder.markdown(full_response)
        filtered_sources = filter_sources_by_answer(full_response, sources)
        filtered_sources = dedupe_sources(filtered_sources)
        if filtered_sources:
            with st.expander("Sources and topics"):
                for s in filtered_sources:
                    st.markdown(f"Source: {s.get('source', 'Unknown')}")
                    st.markdown(f"Topic: {s.get('topic_id', 'N/A')}")
                    words = ", ".join(s.get("topic_words", []))
                    if words:
                        st.markdown(f"Keywords: {words}")
                    st.markdown("---")
        st.session_state.history.append(("assistant", full_response))
        st.session_state.sources_history.append(filtered_sources)