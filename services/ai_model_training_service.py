import numpy as np
import torch
import tensorflow as tf
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from .ml_flows import MLFlows

class AIModelTrainingService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.tf_model = None
        self.ml_flows = MLFlows()

    # ... (keep all existing methods)

    async def run_ml_flows(self):
        return self.ml_flows.run_all_flows()

ai_model_training_service = AIModelTrainingService()

def get_ai_model_training_service() -> AIModelTrainingService:
    return ai_model_training_service
