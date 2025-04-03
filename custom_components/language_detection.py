from typing import Any, Dict, List, Text, Optional
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
import logging

# Import packages without declaring them as dependencies
try:
    from langdetect import detect
except ImportError:
    def detect(text):
        return "en"

try:
    from deep_translator import GoogleTranslator
    translator_available = True
except ImportError:
    translator_available = False

logger = logging.getLogger(__name__)

@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.MESSAGE_TOKENIZER], is_trainable=False
)
class LanguageDetectorTranslator(GraphComponent):
    # Custom component to detect language and translate to English if needed
    
    @classmethod
    def required_packages(cls) -> List[Text]:
        """Return an empty list to bypass package check."""
        return []
    
    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns default config for the component."""
        return {
            "fallback_language": "en",
            "debug": True  # Enables detailed debugging
        }
    
    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "LanguageDetectorTranslator":
        # Creates a new component based on provided config
        return cls(config)
    
    def __init__(self, config: Dict[Text, Any]) -> None:
        """Initialize the language detector translator."""
        self.fallback_language = config.get("fallback_language", "en")
        self.debug = config.get("debug", True)
        
        if self.debug:
            logger.debug("[DEBUG] Initialized LanguageDetectorTranslator component")
            
        # Check if translation is available
        if not translator_available:
            logger.warning("Translation functionality is not available. Only language detection will be performed.")
    
    def process_training_data(
        self, training_data: TrainingData
    ) -> TrainingData:
        """Component doesn't modify the training data."""
        return training_data
    
    def process(self, messages: List[Message]) -> List[Message]:
        """Process incoming messages - detect language and translate if needed."""
        for message in messages:
            self._process_message(message)
        return messages
    
    def _process_message(self, message: Message) -> None:
        """Process an individual message."""
        # Extract text from the message
        text = message.get("text")
        
        if not text:
            logger.warning("No text found in message, skipping language detection")
            return
            
        if self.debug:
            logger.debug(f"[DEBUG] Processing text: '{text}'")
        
        # Detect language
        try:
            detected_language = detect(text)
            if self.debug:
                logger.debug(f"[DEBUG] Detected language: {detected_language}")
            
            # Store language as an entity
            entities = message.get("entities", [])
            entities.append({
                "entity": "detected_language",
                "value": detected_language,
                "start": 0,
                "end": len(text) if text else 0,
                "extractor": "LanguageDetectorTranslator"
            })
            message.set("entities", entities)
            
            if self.debug:
                logger.debug(f"[DEBUG] Added language '{detected_language}' as entity")
            
            # If not English and translator is available, translate to English
            if detected_language != "en" and translator_available:
                original_text = text
                try:
                    # Using deep_translator for translation
                    translator = GoogleTranslator(source=detected_language, target='en')
                    translation = translator.translate(text)
                    
                    if self.debug:
                        logger.debug(f"[DEBUG] Original ({detected_language}): '{original_text}'")
                        logger.debug(f"[DEBUG] Translated (en): '{translation}'")
                    
                    # Save original text as entity too
                    entities.append({
                        "entity": "original_text",
                        "value": original_text,
                        "start": 0,
                        "end": len(text) if text else 0,
                        "extractor": "LanguageDetectorTranslator"
                    })
                    message.set("entities", entities)
                    
                    # Update message text with translation
                    message.set("text", translation)
                    
                except Exception as e:
                    logger.error(f"Translation error: {e}")
                    if self.debug:
                        logger.debug("[DEBUG] Translation failed, keeping original text")
            else:
                if detected_language != "en":
                    logger.warning("Non-English text detected but translator not available. Using original text.")
                if self.debug:
                    logger.debug("[DEBUG] Text will remain in original language")
                
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            if self.debug:
                logger.debug(f"[DEBUG] Language detection failed, assuming {self.fallback_language}")
            
            # Store fallback language as entity
            entities = message.get("entities", [])
            entities.append({
                "entity": "detected_language",
                "value": self.fallback_language,
                "start": 0,
                "end": len(text) if text else 0,
                "extractor": "LanguageDetectorTranslator"
            })
            message.set("entities", entities)