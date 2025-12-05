
class Settings:

    @staticmethod
    def get_prompt():
        return (
            "Eres un asistente en espanol que responde de manera breve y clara "
            "usando solo el contexto proporcionado. Si el contexto no contiene "
            "la informacion necesaria, indica que no la encuentras.\n\n"
            "Contexto:\n{context}\n\n"
            "Pregunta:\n{question}\n\n"
            "Respuesta:"
        )
    
    UC3M_URL = "https://yiyuan.tsc.uc3m.es/api/generate"
    DEFAULT_MODEL = "qwen3:8b"



settings = Settings()