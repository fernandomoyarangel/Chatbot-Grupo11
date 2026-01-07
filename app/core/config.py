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
        2. **NEGATIVE ANSWER:** If the information is not in the context, reply EXACTLY: 
           "I am sorry, I cannot find that information in the available documents." 
           and **DO NOT** include a "Source:" section.
        3. **CONCISENESS:** Be brief and direct.
        
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

    @staticmethod
    def get_surprise_prompt():
        template = """
        SYSTEM: You are a movie trivia generator, NOT a chatbot.
        TASK: Extract ONE interesting fact from the provided text (plot twist, cast detail, or event) and format it as a curiosity.

        STRICT OUTPUT RULES:
        1. Start DIRECTLY with "Did you know that...".
        2. Do NOT use introductory phrases like "Here is a fact" or "Sure!".
        3. Do NOT explain why you chose this fact.
        4. Do NOT include parentheses with meta-comments.
        5. Output ONLY the fact string.

        TEXT DATA:
        {context}

        RESPONSE:
        """
        return PromptTemplate(template=template, input_variables=["context"])

settings = Settings()