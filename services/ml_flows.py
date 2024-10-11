import numpy as np
import torch
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from torch import nn
import torch.optim as optim

class MLFlows:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def numpy_linear_regression(self):
        # Generate sample data
        X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Fit linear regression model
        X_mean, X_std = X_train.mean(), X_train.std()
        y_mean, y_std = y_train.mean(), y_train.std()
        X_train_norm = (X_train - X_mean) / X_std
        y_train_norm = (y_train - y_mean) / y_std

        theta = np.dot(np.linalg.inv(np.dot(X_train_norm.T, X_train_norm)), np.dot(X_train_norm.T, y_train_norm))
        
        # Make predictions
        X_test_norm = (X_test - X_mean) / X_std
        y_pred_norm = np.dot(X_test_norm, theta)
        y_pred = y_pred_norm * y_std + y_mean

        return f"NumPy Linear Regression MSE: {np.mean((y_test - y_pred)**2)}"

    def pytorch_logistic_regression(self):
        # Generate sample data
        X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)

        # Define and train model
        model = nn.Sequential(
            nn.Linear(10, 1),
            nn.Sigmoid()
        ).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters())

        for epoch in range(1000):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs.squeeze(), y_train_tensor)
            loss.backward()
            optimizer.step()

        # Evaluate model
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            predicted = (test_outputs.squeeze() > 0.5).float()
            accuracy = (predicted == y_test_tensor).float().mean()

        return f"PyTorch Logistic Regression Accuracy: {accuracy.item():.4f}"

    def tensorflow_neural_network(self):
        # Generate sample data
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Normalize data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Build model
        model = Sequential([
            Dense(64, activation='relu', input_shape=(20,)),
            Dense(32, activation='relu'),
            Dense(3, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train model
        history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

        # Evaluate model
        _, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
        return f"TensorFlow Neural Network Accuracy: {accuracy:.4f}"

    def numpy_kmeans(self):
        # Generate sample data
        X, _ = make_classification(n_samples=300, n_features=2, n_classes=3, n_clusters_per_class=1, random_state=42)

        # Implement K-means
        k = 3
        centroids = X[np.random.choice(X.shape[0], k, replace=False)]
        
        for _ in range(50):
            distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids

        return f"NumPy K-means Clustering Inertia: {sum((X - centroids[labels])**2).sum()}"

    def pytorch_autoencoder(self):
        # Generate sample data
        X, _ = make_classification(n_samples=1000, n_features=20, n_classes=1, random_state=42)
        X_train, X_test = train_test_split(X, test_size=0.2)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)

        # Define autoencoder
        class Autoencoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(20, 10),
                    nn.ReLU(),
                    nn.Linear(10, 5)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(5, 10),
                    nn.ReLU(),
                    nn.Linear(10, 20)
                )
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded

        model = Autoencoder().to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())

        # Train model
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, X_train_tensor)
            loss.backward()
            optimizer.step()

        # Evaluate model
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, X_test_tensor)

        return f"PyTorch Autoencoder Test Loss: {test_loss.item():.4f}"

    def tensorflow_cnn(self):
        # Generate sample data (simulating image data)
        X = np.random.rand(1000, 28, 28, 1)
        y = np.random.randint(0, 10, 1000)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Build CNN model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train model
        history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=0)

        # Evaluate model
        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        return f"TensorFlow CNN Accuracy: {accuracy:.4f}"

    def numpy_pca(self):
        # Generate sample data
        X, _ = make_classification(n_samples=1000, n_features=20, n_classes=1, random_state=42)

        # Implement PCA
        X_centered = X - np.mean(X, axis=0)
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvectors by decreasing eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Project data onto principal components
        n_components = 10
        X_pca = np.dot(X_centered, eigenvectors[:, :n_components])

        # Calculate explained variance ratio
        explained_variance_ratio = eigenvalues[idx][:n_components] / np.sum(eigenvalues)

        return f"NumPy PCA Explained Variance Ratio: {explained_variance_ratio.sum():.4f}"

    def pytorch_rnn(self):
        # Generate sample data (simulating time series data)
        X = np.random.rand(100, 10, 1)
        y = np.random.randint(0, 2, 100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_tensor = torch.LongTensor(y_test).to(self.device)

        # Define RNN model
        class RNN(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.hidden_size = hidden_size
                self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)
            
            def forward(self, x):
                _, hidden = self.rnn(x)
                out = self.fc(hidden.squeeze(0))
                return out

        model = RNN(1, 32, 2).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

        # Train model
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        # Evaluate model
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == y_test_tensor).float().mean()

        return f"PyTorch RNN Accuracy: {accuracy.item():.4f}"

    def tensorflow_gan(self):
        # Define generator and discriminator
        def make_generator_model():
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Reshape((7, 7, 256)),
                tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
            ])
            return model

        def make_discriminator_model():
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1)
            ])
            return model

        generator = make_generator_model()
        discriminator = make_discriminator_model()

        # Define loss functions and optimizers
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        # Define training step
        @tf.function
        def train_step(images):
            noise = tf.random.normal([images.shape[0], 100])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(noise, training=True)

                real_output = discriminator(images, training=True)
                fake_output = discriminator(generated_images, training=True)

                gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
                disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + cross_entropy(tf.zeros_like(fake_output), fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

            return gen_loss, disc_loss

        # Train GAN (simplified, without real data)
        for epoch in range(50):
            fake_images = tf.random.normal([32, 28, 28, 1])
            gen_loss, disc_loss = train_step(fake_images)

        return f"TensorFlow GAN - Final Generator Loss: {gen_loss:.4f}, Discriminator Loss: {disc_loss:.4f}"

    def numpy_decision_tree(self):
        # Generate sample data
        X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Implement a simple decision tree
        class DecisionTree:
            def __init__(self, max_depth=5):
                self.max_depth = max_depth

            def gini(self, y):
                _, counts = np.unique(y, return_counts=True)
                probabilities = counts / len(y)
                return 1 - np.sum(probabilities**2)

            def split(self, X, y, feature, threshold):
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

            def find_best_split(self, X, y):
                best_gini = float('inf')
                best_feature = None
                best_threshold = None

                for feature in range(X.shape[1]):
                    thresholds = np.unique(X[:, feature])
                    for threshold in thresholds:
                        _, y_left, _, y_right = self.split(X, y, feature, threshold)
                        gini = (len(y_left) * self.gini(y_left) + len(y_right) * self.gini(y_right)) / len(y)
                        if gini < best_gini:
                            best_gini = gini
                            best_feature = feature
                            best_threshold = threshold

                return best_feature, best_threshold

            def build_tree(self, X, y, depth=0):
                if depth == self.max_depth or len(np.unique(y)) == 1:
                    return np.bincount(y).argmax()

                feature, threshold = self.find_best_split(X, y)
                if feature is None:
                    return np.bincount(y).argmax()

                X_left, y_left, X_right, y_right = self.split(X, y, feature, threshold)
                
                return {
                    'feature': feature,
                    'threshold': threshold,
                    'left': self.build_tree(X_left, y_left, depth + 1),
                    'right': self.build_tree(X_right, y_right, depth + 1)
                }

            def fit(self, X, y):
                self.tree = self.build_tree(X, y)

            def predict_sample(self, x, node):
                if isinstance(node, dict):
                    if x[node['feature']] <= node['threshold']:
                        return self.predict_sample(x, node['left'])
                    else:
                        return self.predict_sample(x, node['right'])
                else:
                    return node

            def predict(self, X):
                return np.array([self.predict_sample(x, self.tree) for x in X])

        # Train and evaluate the decision tree
        dt = DecisionTree()
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        accuracy = np.mean(y_pred == y_test)

        return f"NumPy Decision Tree Accuracy: {accuracy:.4f}"

    def run_all_flows(self):
        results = []
        results.append(self.numpy_linear_regression())
        results.append(self.pytorch_logistic_regression())
        results.append(self.tensorflow_neural_network())
        results.append(self.numpy_kmeans())
        results.append(self.pytorch_autoencoder())
        results.append(self.tensorflow_cnn())
        results.append(self.numpy_pca())
        results.append(self.pytorch_rnn())
        results.append(self.tensorflow_gan())
        results.append(self.numpy_decision_tree())
        return "\n".join(results)

ml_flows = MLFlows()
