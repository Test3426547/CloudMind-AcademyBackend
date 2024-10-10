from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

def translate_text(text: str, target_language: str = "fr") -> str:
    # This is a simple example using French as the target language
    # In a real implementation, you would support multiple languages
    translated = translator(text, max_length=40)
    return translated[0]['translation_text']
