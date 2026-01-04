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
</style>
""", unsafe_allow_html=True)

# --- Variables de Entorno ---
API_URL = os.getenv("API_URL", "http://localhost:8000")

# --- Gestión del Estado (Session State) ---
if "history" not in st.session_state:
    st.session_state.history = []

if "language" not in st.session_state:
    st.session_state.language = "English"

def clear_history():
    st.session_state.history = []

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
    st.info(
        "**Info:** Soy tu guionista experto en cine. "
        "Pregúntame sobre directores, tramas, o recomendaciones utilizando RAG."
    )
    st.markdown("© 2025 CineBot Producciones")

# --- Interfaz Principal ---
st.markdown('<div class="main-header">🎬 CineBot Expert</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">"Le haré una oferta que no podrá rechazar... responder sus dudas de cine."</div>', unsafe_allow_html=True)

# Contenedor para el historial de chat
chat_container = st.container()

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

    for role, text in st.session_state.history:
        # Personalizar iconos según rol
        avatar = "👤" if role == "user" else "🎥"
        with st.chat_message(role, avatar=avatar):
            st.markdown(text)

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
                else:
                    error_msg = "Corte! Error" if st.session_state.language == "Español" else "Cut! Error"
                    problem_msg = "El proyeccionista tuvo un problema." if st.session_state.language == "Español" else "The projectionist had a problem."
                    full_response = f"⚠️ **{error_msg} {resp.status_code}:** {problem_msg}\n\n`{resp.text}`"
            except requests.exceptions.RequestException as e:
                error_msg = "Error de producción:" if st.session_state.language == "Español" else "Production error:"
                full_response = f"🚨 **{error_msg}** {e}"

        # Mostrar respuesta final
        message_placeholder.markdown(full_response)
        st.session_state.history.append(("assistant", full_response))
