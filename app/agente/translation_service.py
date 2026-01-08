from functools import lru_cache
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_es_to_en_model():
    """
    Carga y cachea el modelo de traducción Español → Inglés.

    Returns:
        Tuple[MarianTokenizer, MarianMTModel]: Tokenizador y modelo MarianMT.
    """
    try:
        from transformers import MarianMTModel, MarianTokenizer
        
        model_name = "Helsinki-NLP/opus-mt-es-en"
        logger.info(f"Loading translation model: {model_name}")
        
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        
        logger.info(f"Model {model_name} loaded successfully")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading ES->EN model: {e}")
        raise


@lru_cache(maxsize=1)
def get_en_to_es_model():
    """
    Carga y cachea el modelo de traducción Inglés → Español.

    Returns:
        Tuple[MarianTokenizer, MarianMTModel]: Tokenizador y modelo MarianMT.
    """
    try:
        from transformers import MarianMTModel, MarianTokenizer
        
        model_name = "Helsinki-NLP/opus-mt-en-es"
        logger.info(f"Loading translation model: {model_name}")
        
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        
        logger.info(f"Model {model_name} loaded successfully")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading EN->ES model: {e}")
        raise


class TranslationService:  
    """
    Servicio de traducción ES↔EN con caché en memoria.

    Utiliza modelos MarianMT y almacena traducciones repetidas para
    mejorar el rendimiento.
    """ 
    def __init__(self):
        """
        Inicializa las cachés internas de traducción.
        """
        self._es_to_en_cache = {}
        self._en_to_es_cache = {}
    
    def translate_es_to_en(self, text: str, use_cache: bool = True) -> str:
        """
        Traduce texto de Español a Inglés.

        Args:
            text (str): Texto en español a traducir.
            use_cache (bool): Usa la caché interna si está habilitada.

        Returns:
            str: Texto traducido al inglés o el original si ocurre un error.
        """
       
        if not text or not text.strip():
            return text
        
        # Check cache
        if use_cache and text in self._es_to_en_cache:
            logger.debug(f"Using cached translation for: {text[:50]}...")
            return self._es_to_en_cache[text]
        
        try:
            tokenizer, model = get_es_to_en_model()
            

            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            translated = model.generate(**inputs)
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            

            if use_cache:
                self._es_to_en_cache[text] = translated_text
            

            print(f"  ✓ ES→EN: '{text}' → '{translated_text}'")
            logger.info(f"Translated ES->EN: '{text[:50]}...' -> '{translated_text[:50]}...'")
            return translated_text
            
        except Exception as e:
            logger.error(f"Error translating ES->EN: {e}")

            logger.warning("Returning original text due to translation error")
            return text
    
    def translate_en_to_es(self, text: str, use_cache: bool = True) -> str:
        """
        Traduce texto de Inglés a Español.

        Args:
            text (str): Texto en inglés a traducir.
            use_cache (bool): Usa la caché interna si está habilitada.

        Returns:
            str: Texto traducido al español o el original si ocurre un error.
        """
        if not text or not text.strip():
            return text
        

        if use_cache and text in self._en_to_es_cache:
            logger.debug(f"Using cached translation for: {text[:50]}...")
            return self._en_to_es_cache[text]
        
        try:
            tokenizer, model = get_en_to_es_model()
            

            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            translated = model.generate(**inputs)
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            

            if use_cache:
                self._en_to_es_cache[text] = translated_text
            

            print(f"  ✓ EN→ES: '{text}' → '{translated_text}'")
            logger.info(f"Translated EN->ES: '{text[:50]}...' -> '{translated_text[:50]}...'")
            return translated_text
            
        except Exception as e:
            logger.error(f"Error translating EN->ES: {e}")

            logger.warning("Returning original text due to translation error")
            return text
    
    def clear_cache(self):
        """
        Limpia las cachés internas de traducción.
        """
        self._es_to_en_cache.clear()
        self._en_to_es_cache.clear()
        logger.info("Translation caches cleared")
