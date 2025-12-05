
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
