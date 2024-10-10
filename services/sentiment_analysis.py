from transformers import pipeline

sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text: str) -> dict:
    result = sentiment_analyzer(text)[0]
    return {
        "label": result["label"],
        "score": result["score"]
    }
