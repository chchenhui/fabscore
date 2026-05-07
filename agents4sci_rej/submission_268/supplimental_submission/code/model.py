import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

class FairnessAwareLogisticRegression:
    def __init__(self, fairness_penalty=0.1, max_iter=1000):
        self.fairness_penalty = fairness_penalty
        self.max_iter = max_iter
        self.model = None

    def fit(self, X, y, a):
        # Compute class weights for fairness
        unique_groups = np.unique(a)
        group_weights = {}

        for group in unique_groups:
            group_mask = (a == group)
            if np.sum(group_mask) > 0:
                group_weights[group] = len(y) / (2 * np.sum(group_mask))

        # Create sample weights combining class and group rebalancing
        sample_weights = np.ones(len(y))
        for i, group in enumerate(a):
            sample_weights[i] = group_weights[group]

        # Fit logistic regression with adjusted weights
        self.model = LogisticRegression(
            max_iter=self.max_iter,
            random_state=42,
            solver='lbfgs'
        )
        self.model.fit(X, y, sample_weight=sample_weights)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class AdversarialFairnessNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(AdversarialFairnessNet, self).__init__()

        # Main classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Adversarial network (predicts protected attribute from classifier output)
        self.adversary = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y_pred = self.classifier(x)
        a_pred = self.adversary(y_pred)
        return y_pred, a_pred


class AdversarialFairnessClassifier:
    def __init__(self, input_dim, fairness_penalty=0.1, learning_rate=0.01, epochs=100):
        self.fairness_penalty = fairness_penalty
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = AdversarialFairnessNet(input_dim)
        self.criterion = nn.BCELoss()

    def fit(self, X, y, a):
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)
        a_tensor = torch.FloatTensor(a).reshape(-1, 1)

        # Optimizers
        classifier_optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.learning_rate)
        adversary_optimizer = optim.Adam(self.model.adversary.parameters(), lr=self.learning_rate)

        self.model.train()

        for epoch in range(self.epochs):
            # Forward pass
            y_pred, a_pred = self.model(X_tensor)

            # Classification loss
            classification_loss = self.criterion(y_pred, y_tensor)

            # Adversarial loss (adversary tries to predict protected attribute)
            adversarial_loss = self.criterion(a_pred, a_tensor)

            # Update adversary (maximize its ability to predict protected attribute)
            adversary_optimizer.zero_grad()
            adversarial_loss.backward(retain_graph=True)
            adversary_optimizer.step()

            # Update classifier (minimize classification loss, maximize adversary loss)
            classifier_optimizer.zero_grad()
            total_loss = classification_loss - self.fairness_penalty * adversarial_loss
            total_loss.backward()
            classifier_optimizer.step()

            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Classification Loss: {classification_loss.item():.4f}, "
                      f"Adversarial Loss: {adversarial_loss.item():.4f}")

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            y_pred, _ = self.model(X_tensor)
            return (y_pred.numpy() > 0.5).astype(int).flatten()

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            y_pred, _ = self.model(X_tensor)
            probs = y_pred.numpy().flatten()
            return np.column_stack([1 - probs, probs])


class ModelFactory:
    @staticmethod
    def create_baseline_models():
        return {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100),
        }

    @staticmethod
    def create_fairness_models(input_dim, fairness_penalty=0.1):
        return {
            'FairnessAwareLR': FairnessAwareLogisticRegression(fairness_penalty=fairness_penalty),
            'AdversarialNet': AdversarialFairnessClassifier(
                input_dim=input_dim,
                fairness_penalty=fairness_penalty,
                epochs=50  # Reduced for faster training
            )
        }

if __name__ == "__main__":
    # Test the models
    from dataset import SyntheticFairnessDataset

    dataset = SyntheticFairnessDataset(n_samples=500, bias_strength=0.3)
    X_train, X_test, y_train, y_test, a_train, a_test, scaler = dataset.get_train_test_split()

    print("Testing Fairness-Aware Logistic Regression...")
    fair_lr = FairnessAwareLogisticRegression(fairness_penalty=0.1)
    fair_lr.fit(X_train, y_train, a_train)
    predictions = fair_lr.predict(X_test)
    print(f"Accuracy: {np.mean(predictions == y_test):.3f}")

    print("\nTesting Adversarial Fairness Network...")
    adv_net = AdversarialFairnessClassifier(input_dim=X_train.shape[1], fairness_penalty=0.1, epochs=20)
    adv_net.fit(X_train, y_train, a_train)
    predictions = adv_net.predict(X_test)
    print(f"Accuracy: {np.mean(predictions == y_test):.3f}")