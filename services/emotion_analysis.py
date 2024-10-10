from transformers import pipeline

emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

def analyze_emotion(text: str) -> str:
    result = emotion_classifier(text)
    return result[0][0]['label']
