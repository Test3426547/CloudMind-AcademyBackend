import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from typing import List, Dict, Any
import logging
from pydantic import BaseModel, ConfigDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

logger = logging.getLogger(__name__)


class MLFlows:

    def __init__(self):
        self.scaler = StandardScaler()

    def linear_regression_numpy(self,
                                X: np.ndarray,
                                y: np.ndarray,
                                epochs: int = 1000,
                                lr: float = 0.01) -> Dict[str, Any]:
        X = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)

        weights = np.zeros(X_train.shape[1])
        bias = 0

        for _ in range(epochs):
            y_pred = np.dot(X_train, weights) + bias
            d_weights = (1 / len(y_train)) * np.dot(X_train.T,
                                                    (y_pred - y_train))
            d_bias = (1 / len(y_train)) * np.sum(y_pred - y_train)
            weights -= lr * d_weights
            bias -= lr * d_bias

        y_pred_test = np.dot(X_test, weights) + bias
        mse = mean_squared_error(y_test, y_pred_test)

        return {"weights": weights, "bias": bias, "mse": mse}

    def logistic_regression_pytorch(self,
                                    X: np.ndarray,
                                    y: np.ndarray,
                                    epochs: int = 1000,
                                    lr: float = 0.01) -> Dict[str, Any]:
        X = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)

        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)

        model = nn.Linear(X_train.shape[1], 1)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)

        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train.unsqueeze(1))
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            y_pred = (model(X_test) > 0.5).float()
            accuracy = accuracy_score(y_test, y_pred)

        return {"model": model, "accuracy": accuracy}

    def neural_network_pytorch(self,
                                X: np.ndarray,
                                y: np.ndarray,
                                epochs: int = 100,
                                batch_size: int = 32) -> Dict[str, Any]:
        X = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)

        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)

        class NeuralNetwork(nn.Module):
            def __init__(self, input_size):
                super(NeuralNetwork, self).__init__()
                self.fc1 = nn.Linear(input_size, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 1)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = torch.sigmoid(self.fc3(x))
                return x

        model = NeuralNetwork(X_train.shape[1])
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters())

        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred = (model(X_test) > 0.5).float()
            accuracy = accuracy_score(y_test, y_pred)

        return {"model": model, "accuracy": accuracy}

    def convolutional_neural_network_pytorch(
            self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 50,
            batch_size: int = 64) -> Dict[str, Any]:
        X = X.reshape(-1, 1, 28, 28)  # Assuming 28x28 images
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)

        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.LongTensor(y_test)

        class CNN(nn.Module):

            def __init__(self):
                super(CNN, self).__init__()
                self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.fc1 = nn.Linear(64 * 7 * 7, 128)
                self.fc2 = nn.Linear(128, 10)

            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.max_pool2d(x, 2)
                x = torch.relu(self.conv2(x))
                x = torch.max_pool2d(x, 2)
                x = x.view(-1, 64 * 7 * 7)
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        model = CNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred = model(X_test).argmax(dim=1)
            accuracy = accuracy_score(y_test, y_pred)

        return {"model": model, "accuracy": accuracy}

    def recurrent_neural_network_pytorch(
            self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 50,
            batch_size: int = 32) -> Dict[str, Any]:
        X = X.reshape(X.shape[0], X.shape[1], 1)  # Reshape for LSTM input
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)

        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)

        class RNN(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(RNN, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                _, (hidden, _) = self.lstm(x)
                out = self.fc(hidden[-1])
                return torch.sigmoid(out)

        model = RNN(1, 64, 1)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters())

        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred = (model(X_test) > 0.5).float()
            accuracy = accuracy_score(y_test, y_pred)

        return {"model": model, "accuracy": accuracy}

    def gradient_boosting_sklearn(self, X: np.ndarray,
                                  y: np.ndarray) -> Dict[str, Any]:
        from sklearn.ensemble import GradientBoostingClassifier

        X = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)

        model = GradientBoostingClassifier(n_estimators=100,
                                           learning_rate=0.1,
                                           max_depth=3,
                                           random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return {"model": model, "accuracy": accuracy}

    def support_vector_machine_sklearn(self, X: np.ndarray,
                                       y: np.ndarray) -> Dict[str, Any]:
        from sklearn.svm import SVC

        X = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=42)

        model = SVC(kernel='rbf', C=1.0, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return {"model": model, "accuracy": accuracy}

    def k_means_clustering_sklearn(self,
                                   X: np.ndarray,
                                   n_clusters: int = 3) -> Dict[str, Any]:
        from sklearn.cluster import KMeans

        X = self.scaler.fit_transform(X)

        model = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = model.fit_predict(X)

        return {"model": model, "cluster_labels": cluster_labels}

    def principal_component_analysis_sklearn(self,
                                             X: np.ndarray,
                                             n_components: int = 2
                                             ) -> Dict[str, Any]:
        from sklearn.decomposition import PCA

        X = self.scaler.fit_transform(X)

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)

        return {
            "pca": pca,
            "transformed_data": X_pca,
            "explained_variance_ratio": pca.explained_variance_ratio_
        }

    def huggingface_model(self, model_name: str, dataset: List[Dict[str, str]], num_train_epochs: int = 3, learning_rate: float = 2e-5) -> Dict[str, Any]:
        # Convert the dataset to a Hugging Face Dataset
        dataset = Dataset.from_dict({
            "text": [item["text"] for item in dataset],
            "label": [item["label"] for item in dataset]
        })

        # Load pre-trained model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Tokenize the dataset
        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=8,
            learning_rate=learning_rate,
            weight_decay=0.01,
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )

        # Train the model
        trainer.train()

        return {"model": model, "tokenizer": tokenizer}


class AIModelTrainingService:

    def __init__(self):
        self.ml_flows = MLFlows()

    async def train_model(self, model_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if model_type == "huggingface":
                result = self.ml_flows.huggingface_model(
                    model_name=data['model_name'],
                    dataset=data['dataset'],
                    num_train_epochs=data.get('num_train_epochs', 3),
                    learning_rate=data.get('learning_rate', 2e-5)
                )
            else:
                X = np.array(data['features'])
                y = np.array(data['labels'])

                if model_type == "linear_regression":
                    result = self.ml_flows.linear_regression_numpy(X, y)
                elif model_type == "logistic_regression":
                    result = self.ml_flows.logistic_regression_pytorch(X, y)
                elif model_type == "neural_network":
                    result = self.ml_flows.neural_network_pytorch(X, y)
                elif model_type == "cnn":
                    result = self.ml_flows.convolutional_neural_network_pytorch(X, y)
                elif model_type == "rnn":
                    result = self.ml_flows.recurrent_neural_network_pytorch(X, y)
                elif model_type == "gradient_boosting":
                    result = self.ml_flows.gradient_boosting_sklearn(X, y)
                elif model_type == "svm":
                    result = self.ml_flows.support_vector_machine_sklearn(X, y)
                elif model_type == "kmeans":
                    result = self.ml_flows.k_means_clustering_sklearn(X)
                elif model_type == "pca":
                    result = self.ml_flows.principal_component_analysis_sklearn(X)
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")

            return {
                "status": "success",
                "model_type": model_type,
                "result": result
            }
        except Exception as e:
            logger.error(f"Error in train_model: {str(e)}")
            return {"status": "error", "message": str(e)}


ai_model_training_service = AIModelTrainingService()


def get_ai_model_training_service() -> AIModelTrainingService:
    return ai_model_training_service

class HuggingFaceTrainingRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str
    dataset: List[Dict[str, str]]
    num_train_epochs: int = 3
    learning_rate: float = 2e-5
