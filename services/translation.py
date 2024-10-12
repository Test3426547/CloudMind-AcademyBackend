from typing import Dict, Any
import re
import logging

logger = logging.getLogger(__name__)

class TranslationService:
    def __init__(self):
        self.translations = {
            'en': {
                'hello': 'hello',
                'world': 'world',
                'goodbye': 'goodbye',
            },
            'es': {
                'hello': 'hola',
                'world': 'mundo',
                'goodbye': 'adiós',
            },
            'fr': {
                'hello': 'bonjour',
                'world': 'monde',
                'goodbye': 'au revoir',
            },
        }

    async def translate_text(self, text: str, target_language: str, source_language: str = 'auto') -> Dict[str, Any]:
        try:
            if source_language == 'auto':
                source_language = self.detect_language(text)

            words = re.findall(r'\w+', text.lower())
            translated_words = []

            for word in words:
                if word in self.translations[source_language]:
                    translated_word = self.translations[target_language].get(
                        self.translations[source_language][word],
                        word
                    )
                    translated_words.append(translated_word)
                else:
                    translated_words.append(word)

            translated_text = ' '.join(translated_words)

            return {
                "original_text": text,
                "translated_text": translated_text,
                "source_language": source_language,
                "target_language": target_language
            }
        except Exception as e:
            logger.error(f"Error translating text: {str(e)}")
            raise ValueError(f"Translation failed: {str(e)}")

    def detect_language(self, text: str) -> str:
        words = re.findall(r'\w+', text.lower())
        language_scores = {lang: 0 for lang in self.translations.keys()}

        for word in words:
            for lang, translations in self.translations.items():
                if word in translations.values():
                    language_scores[lang] += 1

        detected_language = max(language_scores, key=language_scores.get)
        return detected_language

    async def detect_language_async(self, text: str) -> Dict[str, Any]:
        try:
            detected_language = self.detect_language(text)
            return {
                "language": detected_language,
                "confidence": 1.0  # Since this is a simple implementation, we always return 1.0 as confidence
            }
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")
            raise ValueError(f"Language detection failed: {str(e)}")

translation_service = TranslationService()

def get_translation_service() -> TranslationService:
    return translation_service
