"""
Translation Service for Spanish-English bidirectional translation.

Uses Helsinki-NLP OPUS-MT models for high-quality translation between Spanish and English.
Models are loaded lazily and cached in memory for performance.
"""

from functools import lru_cache
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_es_to_en_model():
    """Load and cache the Spanish to English translation model."""
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
    """Load and cache the English to Spanish translation model."""
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
    def __init__(self):
        """Initialize the translation service."""
        self._es_to_en_cache = {}
        self._en_to_es_cache = {}
    
    def translate_es_to_en(self, text: str, use_cache: bool = True) -> str:
       
        if not text or not text.strip():
            return text
        
        # Check cache
        if use_cache and text in self._es_to_en_cache:
            logger.debug(f"Using cached translation for: {text[:50]}...")
            return self._es_to_en_cache[text]
        
        try:
            tokenizer, model = get_es_to_en_model()
            
            # Tokenize and translate
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            translated = model.generate(**inputs)
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            
            # Cache the result
            if use_cache:
                self._es_to_en_cache[text] = translated_text
            
            # Console output for debugging
            print(f"  ✓ ES→EN: '{text}' → '{translated_text}'")
            logger.info(f"Translated ES->EN: '{text[:50]}...' -> '{translated_text[:50]}...'")
            return translated_text
            
        except Exception as e:
            logger.error(f"Error translating ES->EN: {e}")
            # Fallback: return original text
            logger.warning("Returning original text due to translation error")
            return text
    
    def translate_en_to_es(self, text: str, use_cache: bool = True) -> str:
        if not text or not text.strip():
            return text
        
        # Check cache
        if use_cache and text in self._en_to_es_cache:
            logger.debug(f"Using cached translation for: {text[:50]}...")
            return self._en_to_es_cache[text]
        
        try:
            tokenizer, model = get_en_to_es_model()
            
            # Tokenize and translate
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            translated = model.generate(**inputs)
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            
            # Cache the result
            if use_cache:
                self._en_to_es_cache[text] = translated_text
            
            # Console output for debugging
            print(f"  ✓ EN→ES: '{text}' → '{translated_text}'")
            logger.info(f"Translated EN->ES: '{text[:50]}...' -> '{translated_text[:50]}...'")
            return translated_text
            
        except Exception as e:
            logger.error(f"Error translating EN->ES: {e}")
            # Fallback: return original text
            logger.warning("Returning original text due to translation error")
            return text
    
    def clear_cache(self):
        """Clear translation caches."""
        self._es_to_en_cache.clear()
        self._en_to_es_cache.clear()
        logger.info("Translation caches cleared")
