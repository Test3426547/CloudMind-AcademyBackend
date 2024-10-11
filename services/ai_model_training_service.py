import numpy as np
import torch
import tensorflow as tf
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
import os
import pandas as pd

class AIModelTrainingService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.tf_model = None

    async def train_advanced_model(self, dataset_name=None, num_labels=2, epochs=3, local_dataset_path=None):
        if local_dataset_path:
            dataset = self._load_local_dataset(local_dataset_path)
        elif dataset_name:
            dataset = load_dataset(dataset_name)
        else:
            raise ValueError("Either dataset_name or local_dataset_path must be provided")

        train_texts, train_labels = dataset['train']['text'], dataset['train']['label']
        
        # Split dataset
        train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2)

        # NumPy preprocessing
        train_labels_np = np.array(train_labels)
        val_labels_np = np.array(val_labels)

        # PyTorch model
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

        # Tokenize datasets
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True)
        val_encodings = self.tokenizer(val_texts, truncation=True, padding=True)

        # Create PyTorch datasets
        train_dataset = self.TorchDataset(train_encodings, train_labels_np)
        val_dataset = self.TorchDataset(val_encodings, val_labels_np)

        # Train PyTorch model
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

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        trainer.train()

        # TensorFlow model
        self.tf_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(768,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_labels, activation='softmax')
        ])

        self.tf_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Convert PyTorch embeddings to TensorFlow
        with torch.no_grad():
            train_embeddings = self.model.distilbert(torch.tensor(train_dataset.encodings['input_ids'])).last_hidden_state[:, 0, :].numpy()
            val_embeddings = self.model.distilbert(torch.tensor(val_dataset.encodings['input_ids'])).last_hidden_state[:, 0, :].numpy()

        # Train TensorFlow model
        history = self.tf_model.fit(
            train_embeddings, train_labels_np,
            epochs=epochs,
            validation_data=(val_embeddings, val_labels_np),
            verbose=1
        )

        # Evaluate the models
        torch_eval_results = trainer.evaluate()
        tf_eval_results = self.tf_model.evaluate(val_embeddings, val_labels_np)

        return {
            "message": "Advanced model training completed.",
            "torch_eval_results": torch_eval_results,
            "tf_eval_results": {
                "loss": tf_eval_results[0],
                "accuracy": tf_eval_results[1]
            }
        }

    async def predict_advanced(self, text):
        if self.model is None or self.tokenizer is None or self.tf_model is None:
            raise ValueError("Models not trained. Please train the advanced model first.")

        # PyTorch prediction
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            torch_outputs = self.model(**inputs)
            torch_probs = torch.nn.functional.softmax(torch_outputs.logits, dim=-1)

        # TensorFlow prediction
        tf_inputs = self.model.distilbert(inputs['input_ids']).last_hidden_state[:, 0, :].numpy()
        tf_probs = self.tf_model.predict(tf_inputs)

        # Combine predictions (simple average)
        combined_probs = (torch_probs.numpy() + tf_probs) / 2

        return {
            "torch_prediction": torch_probs.tolist()[0],
            "tf_prediction": tf_probs.tolist()[0],
            "combined_prediction": combined_probs.tolist()[0]
        }

    def _load_local_dataset(self, file_path):
        df = pd.read_csv(file_path)
        return Dataset.from_pandas(df)

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
