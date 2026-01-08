import os
import sys
import requests
import random
import streamlit as st
from langdetect import detect, DetectorFactory, LangDetectException
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ------------------------------------
from app.core.evaluation_logger import log_human_feedback
from app.agente.rag_service import get_llm_model
from langchain_core.messages import HumanMessage

# --- Configuración de la página ---
st.set_page_config(
    page_title="CineBot Expert 🎬",
    page_icon="🍿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- POOL DE PREGUNTAS ---
INITIAL_SUGGESTIONS = {
    "English": [
        "Tell me about some movies that came out in 2008?",
        "In which order should I watch Star Wars?",
        "Recommend me some horror films",
        "Who is Shrek's best friend?",
        "Who are the main characters of Star Wars?"
    ],
    "Español": [
        "¿Háblame de algunas películas que salieron en 2008?",
        "¿En qué orden debería ver Star Wars?",
        "Recomiéndame algunas películas de terror",
        "¿Quién es el mejor amigo de Shrek?",
        "¿Quiénes son los personajes principales de Star Wars?"
    ]
}

# --- DICCIONARIO DE TRADUCCIONES ---
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
        "sub_header": '"I\'m gonna make you an offer you can\'t refuse... answering your movie questions."',
        "placeholder": "Write your cinema questions here...",
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
        "suggestions_label": "💡 You might also ask:",
        "btn_surprise": "🎲 Surprise me!",
        "surprise_prefix": "Did you know...",
        "init_sugg_label": "✨ Or try one of these:",
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
        "placeholder": "Escribe tu pregunta cinematográfica aquí...",
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
        "suggestions_label": "💡 Quizás te interese:",
        "btn_surprise": "🎲 ¡Sorpréndeme!",
        "surprise_prefix": "Sabías que...",
        "init_sugg_label": "✨ O prueba con una de estas:",
    }
}


def get_text(key):
    """Devuelve el texto en el idioma actual."""
    lang = st.session_state.get("language", "English")
    if lang not in TEXTS: lang = "English"
    return TEXTS[lang].get(key, f"MISSING: {key}")


# --- Estilos CSS Personalizados ---
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

    /* ESTILO PARA BOTONES TRANSPARENTES (Feedback) */
    section.main .stButton button {
        background-color: transparent !important;
        border: none !important;
        color: #5D4037 !important; 
        font-size: 1.2rem;
        transition: transform 0.2s, background-color 0.2s;
    }

    section.main .stButton button:hover {
        background-color: rgba(93, 64, 55, 0.1) !important;
        transform: scale(1.1);
        border-radius: 50%;
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

# [CORRECCIÓN] Inicializamos el estado para las sugerencias aleatorias
if "random_suggestions" not in st.session_state:
    st.session_state.random_suggestions = []

if "suggestions_lang" not in st.session_state:
    st.session_state.suggestions_lang = ""

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


def initialize_random_suggestions():
    """Genera 3 sugerencias aleatorias y las guarda en Session State."""
    current_lang = st.session_state.language
    pool = INITIAL_SUGGESTIONS.get(current_lang, INITIAL_SUGGESTIONS["English"])
    num_choices = min(3, len(pool))

    if num_choices > 0:
        st.session_state.random_suggestions = random.sample(pool, num_choices)
        st.session_state.suggestions_lang = current_lang
    else:
        st.session_state.random_suggestions = []


def clear_history():
    st.session_state.history = []
    st.session_state.sources_history = []
    # [IMPORTANTE] Al borrar historial, forzamos nuevas sugerencias llamando a la función
    initialize_random_suggestions()


# --- Inicialización Única al cargar la página ---
# Si la lista está vacía o el idioma cambió respecto a lo guardado, generamos.
if not st.session_state.random_suggestions or st.session_state.suggestions_lang != st.session_state.language:
    initialize_random_suggestions()


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
                                timeout=150
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
    # 1. CABECERA
    col_img, col_txt = st.columns([0.35, 0.65])

    with col_img:
        st.image("https://img.icons8.com/dusk/200/clapperboard.png", width="stretch")

    with col_txt:
        st.markdown(
            f"""<div style='line-height: 1.1; font-weight: bold; color: #FFCC80; padding-top: 5px;'>
            {get_text('sidebar_desc').replace(chr(10), '<br>')}
            </div>""",
            unsafe_allow_html=True
        )

    # 2. SEPARADOR PERSONALIZADO
    st.markdown(
        "<div style='margin-top: -15px; margin-bottom: 10px; border-top: 1px solid #8D6E63;'></div>",
        unsafe_allow_html=True
    )

    # 3. BOTONES DE ACCIÓN
    if st.button(get_text("btn_reset"), width="stretch"):
        clear_history()
        st.rerun()

    if st.button(get_text("btn_surprise"), width="stretch"):
        with st.spinner("🎲 ..."):
            try:
                lang_map = {"English": "english", "Español": "spanish"}
                api_lang = lang_map.get(st.session_state.language, "english")
                resp = requests.post(f"{API_URL}/surprise", json={"language": api_lang}, timeout=60)
                if resp.ok:
                    curiosity = resp.json().get("curiosity", "")
                    prefix = get_text("surprise_prefix")
                    final_text = f"🎲 **{prefix}** {curiosity}" if not curiosity.lower().startswith(
                        prefix.lower()) else f"🎲 **{curiosity}**"
                    st.session_state.history.append(("assistant", final_text))
                    st.session_state.sources_history.append({"sources": [], "suggestions": []})
                    st.rerun()
                else:
                    st.error(f"Err: {resp.text}")
            except Exception as e:
                st.error(f"Conn: {e}")

    # 4. IDIOMA
    st.markdown(f"**{get_text('lang_label')}**")
    st.selectbox("Lang", ["English", "Español"], key="language", label_visibility="collapsed")

    # 5. ANALÍTICA
    st.markdown(f"**{get_text('analytics_title')}**")

    if st.button(get_text("btn_topics_gen"), width="stretch"):
        with st.spinner(get_text("topic_gen_spinner")):
            try:
                lang_map = {"English": "english", "Español": "spanish"}
                requests.post(f"{API_URL}/topics", json={"language": lang_map.get(st.session_state.language)},
                              timeout=300)
                st.success("OK")
            except:
                pass

    if st.button(get_text("btn_topics_vis"), width="stretch"):
        st.session_state.show_topics_modal = True
        with st.spinner(get_text("topic_vis_spinner")):
            try:
                lang_map = {"English": "english", "Español": "spanish"}
                resp = requests.post(f"{API_URL}/topics/visualize",
                                     json={"language": lang_map.get(st.session_state.language)}, timeout=300)
                if resp.ok:
                    res = resp.json()
                    st.session_state.topic_plot_html = res.get("plot_html")
                    st.session_state.topic_topics = res.get("topics")
            except:
                pass

    st.caption(get_text("info_text"))
    st.caption(get_text("footer"))

# --- Interfaz Principal ---
st.markdown(f'<div class="main-header">{get_text("main_title")}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">{get_text("sub_header")}</div>', unsafe_allow_html=True)


def detect_language_with_llm(text: str) -> str:
    """Usa el LLM para clasificar el idioma cuando langdetect falla."""
    try:
        llm = get_llm_model()
        sample = text[:200].replace("\n", " ")
        prompt = f"""
        Classify the language of the following text. 
        Return ONLY one word: 'Español' or 'English'.
        If you are unsure, default to 'English'.

        Text: "{sample}"
        Language:
        """
        response = llm.invoke(prompt)
        detected = response.content.strip().replace(".", "")
        if "Español" in detected or "Spanish" in detected:
            return "Español"
        if "English" in detected or "Ingles" in detected:
            return "English"
        return None
    except Exception as e:
        print(f"Error en detección LLM: {e}")
        return None


def smart_language_detector(text, client_llm=None):
    if not text or len(text.strip()) < 2:
        return None
    try:
        local_code = detect(text)
        if local_code in ['es', 'ca']:
            return "Español"
        if local_code == 'en':
            return "English"
    except LangDetectException:
        pass
    print(f"DEBUG: Usando LLM para detectar idioma de: '{text[:30]}...'")
    return detect_language_with_llm(text)


# --- Input de chat con Detección Automática ---
prompt_to_process = None
user_input = st.chat_input(get_text("placeholder"))

# CASO A: Nuevo mensaje del usuario
if user_input:
    try:
        detected_lang_name = smart_language_detector(user_input, client_llm=None)
        print(detected_lang_name)
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

if prompt_to_process:
    st.session_state.show_topics_modal = False


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
    # --- LOGICA: PANTALLA INICIAL (Historial Vacío) ---
    # [CORRECCIÓN APLICADA AQUÍ]
    # Se añade 'and not prompt_to_process' para que desaparezcan
    # si hay una pregunta esperando ser respondida.
    if not st.session_state.history and not prompt_to_process:
        st.markdown(
            f"""
            <div style="text-align: center; margin-top: 50px; color: #8D6E63;">
                <h3>{get_text("empty_chat_title")}</h3>
                <p>{get_text("empty_chat_desc")}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        if st.session_state.random_suggestions:
            st.write("")
            st.markdown(
                f"<div style='text-align:center; color:#8D6E63; margin-bottom:15px;'><i>{get_text('init_sugg_label')}</i></div>",
                unsafe_allow_html=True)

            col_s1, col_s2, col_s3 = st.columns(3)
            cols_list = [col_s1, col_s2, col_s3]

            for idx, suggestion in enumerate(st.session_state.random_suggestions):
                if cols_list[idx].button(suggestion, key=f"init_sugg_{idx}"):
                    st.session_state.saved_prompt = suggestion
                    st.session_state.pending_language = st.session_state.language
                    st.rerun()

    # --- LOGICA: HISTORIAL EXISTENTE ---
    assistant_idx = 0

    for i, (role, text) in enumerate(st.session_state.history):
        avatar = "👤" if role == "user" else "🎥"
        with st.chat_message(role, avatar=avatar):
            st.markdown(text)

            if role == "assistant":
                sources = []
                suggestions = []
                if assistant_idx < len(st.session_state.sources_history):
                    data = st.session_state.sources_history[assistant_idx]
                    if isinstance(data, dict):
                        sources = data.get("sources", [])
                        suggestions = data.get("suggestions", [])
                    elif isinstance(data, list):
                        sources = data

                if sources:
                    render_sources_with_summary(sources, unique_key_suffix=f"hist_{i}")

                if suggestions:
                    st.markdown(f"**{get_text('suggestions_label')}**")
                    scols = st.columns(len(suggestions))
                    for idx_s, sug in enumerate(suggestions):
                        if scols[idx_s].button(sug, key=f"sugg_{i}_{idx_s}"):
                            st.session_state.saved_prompt = sug
                            st.session_state.pending_language = st.session_state.language
                            st.rerun()

                st.write("---")
                col_spacer, col_like, col_dislike = st.columns([14, 1, 1])

                with col_like:
                    if st.button("👍", key=f"like_hist_{i}"):
                        log_human_feedback(st.session_state.history[i - 1][1] if i > 0 else "", text, rating=1)
                        st.toast("Guardado: 👍", icon="✅")

                with col_dislike:
                    if st.button("👎", key=f"dislike_hist_{i}"):
                        log_human_feedback(st.session_state.history[i - 1][1] if i > 0 else "", text, rating=0)
                        st.toast("Guardado: 👎", icon="📝")

                assistant_idx += 1

# --- Procesamiento del Mensaje ---
if prompt_to_process:
    st.session_state.show_topics_modal = False
    prompt = prompt_to_process

    st.session_state.history.append(("user", prompt))
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🎥"):
        message_placeholder = st.empty()
        full_response = ""
        full_answer_obj = {}
        sources = []
        suggestions = []
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
                    full_answer_obj = data.get("answer", {})
                    full_response = full_answer_obj.get("content", "")
                    sources = full_answer_obj.get("sources", [])
                else:
                    full_response = f"⚠️ **{get_text('error_cut')} {resp.status_code}:** {get_text('error_proj')}\n\n`{resp.text}`"
            except requests.exceptions.RequestException as e:
                full_response = f"🚨 **Error:** {e}"

        message_placeholder.markdown(full_response)

        filtered_sources = dedupe_sources(sources)
        full_answer_obj["sources"] = filtered_sources

        if filtered_sources:
            render_sources_with_summary(filtered_sources, unique_key_suffix="curr")

        suggestions = full_answer_obj.get("suggestions", [])
        if suggestions:
            st.markdown(f"**{get_text('suggestions_label')}**")
            cols = st.columns(len(suggestions))
            for i, suggestion in enumerate(suggestions):
                if cols[i].button(suggestion, key=f"sugg_curr_{i}"):
                    st.session_state.saved_prompt = suggestion
                    st.session_state.pending_language = st.session_state.language
                    st.rerun()

        st.write("---")
        col_spacer_curr, col_like_curr, col_dislike_curr = st.columns([14, 1, 1])

        with col_like_curr:
            if st.button("👍", key="like_current"):
                log_human_feedback(prompt, full_response, rating=1)
                st.toast("Guardado: 👍", icon="✅")

        with col_dislike_curr:
            if st.button("👎", key="dislike_current"):
                log_human_feedback(prompt, full_response, rating=0)
                st.toast("Guardado: 👎", icon="📝")

        st.session_state.history.append(("assistant", full_response))
        st.session_state.sources_history.append(full_answer_obj)