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

    async def train_with_huggingface(self, model_name, dataset_name, num_labels):
        try:
            # Load pre-trained model and tokenizer
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Load dataset
            dataset = load_dataset(dataset_name)

            # Tokenize dataset
            def tokenize_function(examples):
                return self.tokenizer(examples["text"], padding="max_length", truncation=True)

            tokenized_datasets = dataset.map(tokenize_function, batched=True)

            # Split dataset
            train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
            eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(200))

            # Define training arguments
            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=3,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir="./logs",
            )

            # Initialize Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset
            )

            # Train the model
            trainer.train()

            # Evaluate the model
            eval_results = trainer.evaluate()

            return {
                "message": "Model trained successfully with Hugging Face",
                "eval_results": eval_results
            }
        except Exception as e:
            return {"error": f"Error during Hugging Face training: {str(e)}"}

    async def run_ml_flows(self):
        return self.ml_flows.run_all_flows()

ai_model_training_service = AIModelTrainingService()

def get_ai_model_training_service() -> AIModelTrainingService:
    return ai_model_training_service
