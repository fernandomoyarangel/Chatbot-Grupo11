import os
import requests
import streamlit as st
from langdetect import detect, DetectorFactory, LangDetectException

# --- Configuración de la página ---
st.set_page_config(
    page_title="CineBot Expert 🎬",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- DICCIONARIO DE TRADUCCIONES (Internacionalización) ---
TEXTS = {
    "English": {
        "sidebar_title": "Box Office & Settings",
        "sidebar_desc": "Manage your movie session.",
        "btn_reset": "🗑️ Cut! (Clear History)",
        "lang_label": "🌐 **Language**",
        "analytics_title": "Analytics",
        "btn_topics_gen": "Generate Topics (BERTopic)",
        "btn_topics_vis": "Visualize Topics",
        "info_text": "**Info:** I am your expert movie screenwriter. Ask me about directors, plots, or recommendations using RAG.",
        "footer": "© 2025 CineBot Productions",
        "main_title": "🎬 CineBot Expert",
        "sub_header": '"I\'m gonna make him an offer he can\'t refuse... answering his movie questions."',
        "placeholder": "Write your script here...",
        "spinner": "Consulting the film library...",
        "error_cut": "Cut! Error",
        "error_proj": "The projectionist had a problem.",
        "sources_expander": "Sources and Topics",
        "btn_summarize": "📝 Summarize",
        "btn_reading": "Reading document...",
        "modal_title": "Summary",
        "keywords": "Keywords",
        "empty_chat_title": "🍿 Silence on the set!",
        "empty_chat_desc": "What movie or fact are we looking for today?",
        "topic_gen_spinner": "Modeling topics...",
        "topic_vis_spinner": "Generating topic chart...",
        "suggestions_label": "💡 You might also ask:"
    },
    "Español": {
        "sidebar_title": "Taquilla & Ajustes",
        "sidebar_desc": "Gestiona tu sesión de cine.",
        "btn_reset": "🗑️ ¡Corten! (Borrar Historial)",
        "lang_label": "🌐 **Idioma / Language**",
        "analytics_title": "Analítica",
        "btn_topics_gen": "Generar tópicos (BERTopic)",
        "btn_topics_vis": "Visualizar tópicos",
        "info_text": "**Info:** Soy tu guionista experto en cine. Pregúntame sobre directores, tramas, o recomendaciones utilizando RAG.",
        "footer": "© 2025 CineBot Producciones",
        "main_title": "🎬 CineBot Expert",
        "sub_header": '"Le haré una oferta que no podrá rechazar... responder sus dudas de cine."',
        "placeholder": "Escribe tu guion aquí...",
        "spinner": "Consultando la filmoteca...",
        "error_cut": "¡Corte! Error",
        "error_proj": "El proyeccionista tuvo un problema.",
        "sources_expander": "Fuentes y Tópicos",
        "btn_summarize": "📝 Resumir",
        "btn_reading": "Leyendo documento...",
        "modal_title": "Resumen",
        "keywords": "Palabras clave",
        "empty_chat_title": "🍿 ¡Silencio en el set!",
        "empty_chat_desc": "¿Qué película o dato cinematográfico buscamos hoy?",
        "topic_gen_spinner": "Modelando tópicos...",
        "topic_vis_spinner": "Generando gráfico de tópicos...",
        "suggestions_label": "💡 Quizás te interese:"
    }
}


def get_text(key):
    """Devuelve el texto en el idioma actual."""
    lang = st.session_state.get("language", "English")
    if lang not in TEXTS: lang = "English"
    return TEXTS[lang].get(key, f"MISSING: {key}")


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
        color: #000000;
    }

    .stChatMessage p, .stChatMessage div, .stChatMessage span {
        color: #000000 !important;
    }

    /* Mensaje del Usuario */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #FFF3E0;
        border-left: 5px solid #E67E22;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #4E342E;
        color: #EFEBE9;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #FFCC80;
    }
    [data-testid="stSidebar"] p {
        color: #D7CCC8;
    }

    /* Input de chat */
    .stChatInputContainer {
        padding-bottom: 20px;
    }

    /* Botones de sugerencias */
    .stButton button {
        border-radius: 20px;
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
    div[role="dialog"] > div{
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }
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

# Lógica para aplicar cambio de idioma pendiente (RERUN TRICK)
if "pending_language" in st.session_state:
    st.session_state.language = st.session_state.pending_language
    del st.session_state.pending_language

# Estados para Tópicos
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


# --- Funciones Auxiliares ---
def show_summary_modal(title, content):
    if hasattr(st, "dialog"):
        @st.dialog(title)
        def _dialog():
            st.write(content)

        _dialog()
    elif hasattr(st, "modal"):
        with st.modal(title):
            st.write(content)
    else:
        st.info(f"**{title}**\n\n{content}")


def render_sources_with_summary(sources, unique_key_suffix):
    """Muestra fuentes con botón de resumen usando el diccionario de idiomas."""
    if not sources:
        return

    with st.expander(get_text("sources_expander")):
        for i, source in enumerate(sources):
            col1, col2 = st.columns([0.75, 0.25])
            src_name = source.get("source", "Unknown")

            with col1:
                st.markdown(f"**Source:** {src_name}")
                st.caption(f"Topic: {source.get('topic_id', 'N/A')}")

            with col2:
                # Texto del botón desde el diccionario
                btn_label = get_text("btn_summarize")
                btn_key = f"btn_sum_{unique_key_suffix}_{i}"
                if st.button(btn_label, key=btn_key):
                    with st.spinner(get_text("btn_reading")):
                        try:
                            lang_map = {"English": "english", "Español": "spanish"}
                            api_lang = lang_map.get(st.session_state.language, "english")

                            resp = requests.post(
                                f"{API_URL}/summary",
                                json={"filename": src_name, "language": api_lang},
                                timeout=60
                            )
                            if resp.ok:
                                summary = resp.json().get("summary", "")
                                show_summary_modal(f"{get_text('modal_title')}: {src_name}", summary)
                            else:
                                st.error(f"Error: {resp.text}")
                        except Exception as e:
                            st.error(f"Error: {e}")

            words = ", ".join(source.get("topic_words", []))
            if words:
                st.markdown(f"*{get_text('keywords')}:* {words}")
            st.divider()


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


# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/dusk/200/clapperboard.png", output_format="PNG")
    st.title(get_text("sidebar_title"))
    st.markdown("---")
    st.write(get_text("sidebar_desc"))

    if st.button(get_text("btn_reset"), use_container_width=True):
        clear_history()
        st.rerun()

    st.markdown("---")

    # Language selector (AUTOMÁTICO con key="language")
    st.write(get_text("lang_label"))
    st.selectbox(
        "Selecciona el idioma",
        ["English", "Español"],
        key="language",  # Conecta con st.session_state.language
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.write(get_text("analytics_title"))

    # Botón Generar Tópicos
    if st.button(get_text("btn_topics_gen"), use_container_width=True):
        with st.spinner(get_text("topic_gen_spinner")):
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
                    st.caption(f"Temas: {result['topic_count']} | Documentos: {result['document_count']}")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except requests.exceptions.RequestException as exc:
                st.error(f"Error API: {exc}")

    # Botón Visualizar Tópicos
    if st.button(get_text("btn_topics_vis"), use_container_width=True):
        st.session_state.show_topics_modal = True
        st.session_state.topic_error = None
        st.session_state.topic_plot_html = None
        st.session_state.topic_topics = None
        with st.spinner(get_text("topic_vis_spinner")):
            try:
                lang_map = {"English": "english", "Español": "spanish"}
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
                st.session_state.topic_error = f"Error API: {exc}"

    st.markdown("---")
    st.info(get_text("info_text"))
    st.markdown(get_text("footer"))

# --- Interfaz Principal ---
st.markdown(f'<div class="main-header">{get_text("main_title")}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">{get_text("sub_header")}</div>', unsafe_allow_html=True)
def smart_language_detector(text, client_llm=None):
    """
    Devuelve "Español", "English" o None si no está seguro.
    Integra langdetect y fallback a LLM.
    """
    if not text or len(text.strip()) < 2:
        return None

    # --- 1. INTENTO LOCAL (Rápido) ---
    try:
        local_code = detect(text)
    except LangDetectException:
        local_code = None

    # Mapeo directo: Si es local y seguro, retornamos ya el nombre final
    if local_code in ['es', 'ca']:
        return "Español"
    if local_code == 'en':
        return "English"
    return None
    # --- 2. INTENTO LLM ---
    # Si llegamos aquí, es porque langdetect falló, dio un idioma raro (hr, it...)
    # o el texto es muy corto.

    # -- AQUÍ LLAMAS A TU LLM --
    # Ejemplo conceptual (descomenta y adapta a tu cliente real):
    # try:
    #     prompt = f"Clasifica el idioma de: '{text}'. Responde solo 'es' o 'en'. Si es catalan responde 'es'."
    #     response = client_llm.chat.completions.create(..., messages=[...])
    #     llm_code = response.choices[0].message.content.strip().lower()
    # except:
    #     llm_code = "error"


    # Mapeo de la respuesta del LLM
    if llm_code in ['es', 'ca', 'spanish', 'español']:
        return "Español"
    if llm_code in ['en', 'english']:
        return "English"

    return None

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
                # Forzar Plotly responsive
                html = f"""
                <div style="width:100%; height:100%; margin:0; padding:0;">
                <style>
                    html, body {{ margin:0; padding:0; width:100%; height:100%; overflow:hidden; }}
                    .plotly-graph-div {{ width:100% !important; height:100% !important; }}
                </style>
                {html}
                <script>
                    const resizePlot = () => {{
                    const gd = document.querySelector('.plotly-graph-div');
                    if (gd && window.Plotly) {{ window.Plotly.Plots.resize(gd); }}
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
                st.dataframe(st.session_state.topic_topics, use_container_width=True, height=420)
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
        render_topics_modal()

# Contenedor para el historial de chat
chat_container = st.container()

with chat_container:
    if not st.session_state.history:
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 50px; color: #8D6E63;">
                <h3>{get_text("empty_chat_title")}</h3>
                <p>{get_text("empty_chat_desc")}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    assistant_idx = 0
    for role, text in st.session_state.history:
        avatar = "👤" if role == "user" else "🎥"
        with st.chat_message(role, avatar=avatar):
            st.markdown(text)
            if role == "assistant":
                sources = []
                suggestions = []  # [NEW] Variable para sugerencias

                # Acceder a la info guardada en history
                if assistant_idx < len(st.session_state.sources_history):
                    # [MODIFIED] Ahora se asume que sources_history guarda todo el objeto 'answer'
                    data_block = st.session_state.sources_history[assistant_idx]

                    # Compatibilidad hacia atrás si antes solo se guardaban listas
                    if isinstance(data_block, list):
                        sources = data_block
                    elif isinstance(data_block, dict):
                        sources = data_block.get("sources", [])
                        suggestions = data_block.get("suggestions", [])

                assistant_idx += 1

                if sources:
                    render_sources_with_summary(sources, unique_key_suffix=f"hist_{assistant_idx}")

                # [NEW] Renderizar sugerencias como botones
                if suggestions:
                    st.markdown(f"**{get_text('suggestions_label')}**")
                    # Usamos columnas para que no ocupen tanto espacio vertical
                    cols = st.columns(len(suggestions))
                    for i, suggestion in enumerate(suggestions):
                        # Clave única obligatoria para cada botón
                        if cols[i].button(suggestion, key=f"sugg_{assistant_idx}_{i}"):
                            # Al hacer click, guardamos el prompt y recargamos
                            st.session_state.saved_prompt = suggestion
                            st.session_state.pending_language = st.session_state.language
                            st.rerun()

# --- Input de chat con Detección Automática ---
prompt_to_process = None
user_input = st.chat_input(get_text("placeholder"))

# CASO A: Nuevo mensaje del usuario
if user_input:
    try:
        detected_lang_name = smart_language_detector(user_input, client_llm=None)
        if detected_lang_name:
            target_lang = detected_lang_name
        else:
            target_lang = st.session_state.language
        if st.session_state.language != target_lang:
            st.session_state.pending_language = target_lang
            st.session_state.saved_prompt = user_input
            st.rerun()
        else:
            prompt_to_process = user_input
    except (LangDetectException, ImportError):
        prompt_to_process = user_input

# CASO B: Mensaje guardado tras recarga
elif "saved_prompt" in st.session_state:
    prompt_to_process = st.session_state.saved_prompt
    del st.session_state.saved_prompt

# --- Procesamiento del Mensaje ---
if prompt_to_process:
    prompt = prompt_to_process

    st.session_state.history.append(("user", prompt))
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🎥"):
        message_placeholder = st.empty()
        full_response = ""
        full_answer_obj = {}  # [NEW]

        with st.spinner(get_text("spinner")):
            sources = []
            try:
                lang_map = {"English": "english", "Español": "spanish"}
                api_lang = lang_map.get(st.session_state.language, "english")

                resp = requests.post(
                    f"{API_URL}/chat",
                    json={"question": prompt, "language": api_lang},
                    timeout=120,
                )
                if resp.ok:
                    data = resp.json()
                    full_answer_obj = data.get("answer", {})  # [NEW] Guardamos todo el objeto
                    full_response = full_answer_obj.get("content", "")
                    sources = full_answer_obj.get("sources", [])
                else:
                    full_response = f"⚠️ **{get_text('error_cut')} {resp.status_code}:** {get_text('error_proj')}\n\n`{resp.text}`"
            except requests.exceptions.RequestException as e:
                full_response = f"🚨 **Error:** {e}"

        message_placeholder.markdown(full_response)

        filtered_sources = dedupe_sources(sources)

        # Actualizamos fuentes filtradas en el objeto antes de guardar
        full_answer_obj["sources"] = filtered_sources
        # suggestions ya viene en full_answer_obj["suggestions"]

        if filtered_sources:
            render_sources_with_summary(filtered_sources, unique_key_suffix="curr")

        # [NEW] Mostrar sugerencias en la respuesta actual
        suggestions = full_answer_obj.get("suggestions", [])
        if suggestions:
            st.markdown(f"**{get_text('suggestions_label')}**")
            cols = st.columns(len(suggestions))
            for i, suggestion in enumerate(suggestions):
                # OJO: key 'curr' para diferenciar del historial
                if cols[i].button(suggestion, key=f"sugg_curr_{i}"):
                    st.session_state.saved_prompt = suggestion
                    st.session_state.pending_language = st.session_state.language
                    st.rerun()

        st.session_state.history.append(("assistant", full_response))

        # [MODIFIED] Guardamos TODO el objeto (sources + suggestions) en el historial
        st.session_state.sources_history.append(full_answer_obj)


