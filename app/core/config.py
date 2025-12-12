from langchain_core.prompts import PromptTemplate
class Settings:

    @staticmethod
    def get_prompt():
        # --- PROMPT EN INGLÉS ---
        template_english = """
        SYSTEM INSTRUCTIONS:
        You are an expert movie assistant. Answer the user's question based EXCLUSIVELY on the context provided below.
        
        STRICT RULES:
        1. **ONLY CONTEXT:** Do not use prior knowledge. If the answer is not explicitly in the text below, DO NOT invent it.
        2. **NEGATIVE ANSWER:** If the information is not in the context, reply exactly: "I am sorry, I cannot find that information in the available documents."
        3. **CITATIONS:** At the end of your answer, you MUST explicitly list the source filename using the format: "Source: [filename]" as it appears in the context.
        4. **CONCISENESS:** Be brief and direct.
        
        RETRIEVED CONTEXT:
        --------------------
        {context}
        --------------------
        
        USER QUESTION: {question}
        
        ANSWER:
        """
        
        return PromptTemplate(
            template=template_english, 
            input_variables=["context", "question"]
        )
            
    UC3M_URL = "https://yiyuan.tsc.uc3m.es/api/generate"
    DEFAULT_MODEL = "qwen3:8b"



settings = Settings()