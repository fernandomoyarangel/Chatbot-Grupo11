import os
import ast
import requests
import streamlit as st

# --- 1. CONFIGURACIÓN INICIAL ---
st.set_page_config(
    page_title="CineBot RAG",
    page_icon="🎬",
    layout="centered"
)

# Definir URL del backend (con fallback seguro)
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Título y descripción
st.title("🎬 CineBot: Experto en Películas")
st.markdown("Pregunta sobre tramas, años o detalles de las 42k películas de la base de datos.")

# --- 2. GESTIÓN DEL ESTADO (HISTORIAL) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. MOSTRAR HISTORIAL AL INICIO (RENDERIZADO) ---
# Iteramos sobre el historial para pintar las burbujas de chat anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. INTERFAZ DE INPUT Y LÓGICA ---
# st.chat_input crea la caja de texto fija abajo y maneja el "Enter" automáticamente
if prompt := st.chat_input("Escribe tu pregunta sobre una película..."):
    
    # A. Mostrar mensaje del usuario inmediatamente
    st.chat_message("user").markdown(prompt)
    # Guardar en historial
    st.session_state.messages.append({"role": "user", "content": prompt})

    # B. Llamar al Backend y mostrar respuesta
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # Placeholder para simular escritura o mostrar carga
        message_placeholder.markdown("⏳ *Consultando base de datos...*")
        
        try:
            # Llamada al backend
            resp = requests.post(
                f"{API_URL}/chat",
                json={"question": prompt},
                timeout=30
            )
            
            if resp.ok:
                data = resp.json()
                raw_response = data.get("answer", "")

                # LÓGICA DE LIMPIEZA "NUCLEAR"
                texto_final = raw_response

                # 1. Si es un diccionario real (JSON puro)
                if isinstance(raw_response, dict):
                    
                    texto_final = raw_response["content"]
                    print(str(raw_response))
                
                # 3. Mostrar el texto limpio
                message_placeholder.markdown(texto_final)
                
                # Guardar en historial
                st.session_state.messages.append({"role": "assistant", "content": texto_final})

                # (Opcional) Fuentes
                if "sources" in data:
                    with st.expander("Ver fuentes consultadas"):
                        st.json(data["sources"])

            else:
                error_msg = f"⚠️ Error del servidor: {resp.status_code}"
                message_placeholder.error(error_msg)
                
        except requests.exceptions.ConnectionError:
            message_placeholder.error("❌ No se pudo conectar con el servidor (Backend caído).")
        except Exception as e:
            message_placeholder.error(f"❌ Error inesperado: {str(e)}")