from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import torch
import os
import pandas as pd
import numpy as np

class AIModelTrainingService:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    async def train_model(self, dataset_name=None, num_labels=2, model_name="distilbert-base-uncased", epochs=3, local_dataset_path=None):
        if local_dataset_path:
            dataset = self._load_local_dataset(local_dataset_path)
        elif dataset_name:
            dataset = load_dataset(dataset_name)
        else:
            raise ValueError("Either dataset_name or local_dataset_path must be provided")

        train_texts, train_labels = dataset['train']['text'], dataset['train']['label']
        
        # Split dataset
        train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        # Tokenize datasets
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True)
        val_encodings = self.tokenizer(val_texts, truncation=True, padding=True)

        # Create torch datasets
        train_dataset = AIModelTrainingService.TorchDataset(train_encodings, train_labels)
        val_dataset = AIModelTrainingService.TorchDataset(val_encodings, val_labels)

        # Set up training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=100,
            load_best_model_at_end=True,
        )

        # Create Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        # Train the model
        trainer.train()

        # Save the model
        self.model.save_pretrained("./trained_model")
        self.tokenizer.save_pretrained("./trained_model")

        # Evaluate the model
        eval_results = trainer.evaluate()

        return {
            "message": "Model training completed and saved.",
            "eval_results": eval_results
        }

    async def predict(self, text):
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not trained. Please train the model first.")

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs.tolist()[0]

    def _load_local_dataset(self, file_path):
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise ValueError("Unsupported file format. Please use CSV or JSON.")

        dataset = Dataset.from_pandas(df)
        return dataset.train_test_split(test_size=0.2)

    class TorchDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

ai_model_training_service = AIModelTrainingService()

def get_ai_model_training_service() -> AIModelTrainingService:
    return ai_model_training_service
