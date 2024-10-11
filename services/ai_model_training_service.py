import numpy as np
import torch
import tensorflow as tf
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from .ml_flows import MLFlows
from fastapi import HTTPException
import logging
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HuggingFaceTrainingRequest(BaseModel):
    model_name: str = Field(..., min_length=1, max_length=100)
    dataset_name: str = Field(..., min_length=1, max_length=100)
    num_labels: int = Field(..., ge=2, le=100)

    @validator('model_name', 'dataset_name')
    def validate_string_fields(cls, v):
        if not v.strip():
            raise ValueError("Field cannot be empty or just whitespace")
        return v

class AIModelTrainingService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.tf_model = None
        self.ml_flows = MLFlows()

    async def train_with_huggingface(self, request: HuggingFaceTrainingRequest) -> Dict[str, Any]:
        try:
            # Load pre-trained model and tokenizer
            self.model = AutoModelForSequenceClassification.from_pretrained(request.model_name, num_labels=request.num_labels)
            self.tokenizer = AutoTokenizer.from_pretrained(request.model_name)

            # Load dataset
            try:
                dataset = load_dataset(request.dataset_name)
            except Exception as e:
                logger.error(f"Error loading dataset: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Failed to load dataset: {str(e)}")

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
            train_result = trainer.train()

            # Evaluate the model
            eval_results = trainer.evaluate()

            return {
                "message": "Model trained successfully with Hugging Face",
                "train_results": train_result.metrics,
                "eval_results": eval_results
            }
        except Exception as e:
            logger.error(f"Error during Hugging Face training: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error during Hugging Face training: {str(e)}")

    async def run_ml_flows(self) -> str:
        try:
            return self.ml_flows.run_all_flows()
        except Exception as e:
            logger.error(f"Error running ML flows: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error running ML flows: {str(e)}")

ai_model_training_service = AIModelTrainingService()

def get_ai_model_training_service() -> AIModelTrainingService:
    return ai_model_training_service
