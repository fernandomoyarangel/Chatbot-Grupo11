
class Settings:

    @staticmethod
    def get_prompt():
        return (
            "Eres un asistente en español. Responde de forma breve, directa y solo con la "
            "información del contexto. Si falta info, di explícitamente que no está en el "
            "contexto.\n\n"
            "Contexto:\n{context}\n\n"
            "Pregunta:\n{question}\n\n"
            "Instrucciones:\n"
            "- Sé conciso (2-4 frases). No inventes.\n"
            "- Cita SIEMPRE AL FINAL DE LA RESPUESTA de qué documentos proviene el contexto.\n"
            "- Si el contexto no contiene la respuesta, di que no la encuentras en el contexto.\n\n"
            "Respuesta:"
        )
            
    UC3M_URL = "https://yiyuan.tsc.uc3m.es/api/generate"
    DEFAULT_MODEL = "qwen3:8b"



settings = Settings()