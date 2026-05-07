"""
CNN-BiGRU Phishing Detector
Based on "Advancing Phishing Email Detection: A Comparative Study of Deep Learning Models"
Sensors 2024, 24(7), 2077
https://www.mdpi.com/1424-8220/24/7/2077

Implements 1D-CNN with Bidirectional GRU for phishing email detection
"""

import numpy as np
import logging
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class CNNBiGRUModel(nn.Module):
    """
    1D-CNN with Bidirectional GRU model for phishing detection
    Architecture based on the 2024 Sensors paper
    """

    def __init__(self, vocab_size=10000, embedding_dim=128, max_length=500):
        super(CNNBiGRUModel, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 1D Convolutional layers
        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)

        # Pooling
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)

        # Bidirectional GRU
        self.bigru = nn.GRU(
            input_size=32,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)  # 64*2 for bidirectional
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # Binary classification

    def forward(self, x):
        # Embedding
        x = self.embedding(x)  # [batch, seq_len, embedding_dim]

        # Transpose for conv1d
        x = x.transpose(1, 2)  # [batch, embedding_dim, seq_len]

        # CNN layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # Transpose back for GRU
        x = x.transpose(1, 2)  # [batch, seq_len, channels]

        # BiGRU
        x, _ = self.bigru(x)

        # Use the last hidden state
        x = x[:, -1, :]  # [batch, hidden_size*2]

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class CNNBiGRUDetector:
    """
    CNN-BiGRU based phishing email detector
    Implements the approach from the 2024 Sensors paper
    """

    def __init__(self, vocab_size=10000, max_length=500):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.model = None
        self.word_to_idx = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the CNN-BiGRU model"""
        self.model = CNNBiGRUModel(
            vocab_size=self.vocab_size,
            embedding_dim=128,
            max_length=self.max_length
        )
        self.model.to(self.device)
        logger.info(f"CNN-BiGRU model initialized on {self.device}")

    def _build_vocabulary(self, texts: List[str]):
        """Build vocabulary from training texts"""
        word_freq = {}

        for text in texts:
            words = self._tokenize(text)
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and take top vocab_size words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        # Reserve indices for special tokens
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}

        for word, _ in sorted_words[:self.vocab_size - 2]:
            self.word_to_idx[word] = len(self.word_to_idx)

        logger.info(f"Vocabulary built with {len(self.word_to_idx)} words")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        import re
        # Convert to lowercase and split on non-alphanumeric
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words

    def _text_to_indices(self, text: str) -> List[int]:
        """Convert text to indices"""
        words = self._tokenize(text)[:self.max_length]
        indices = []

        for word in words:
            if word in self.word_to_idx:
                indices.append(self.word_to_idx[word])
            else:
                indices.append(self.word_to_idx['<UNK>'])

        # Padding
        while len(indices) < self.max_length:
            indices.append(self.word_to_idx['<PAD>'])

        return indices[:self.max_length]

    def train(self, train_data: List[Dict], val_data: List[Dict] = None):
        """Train the CNN-BiGRU model"""
        logger.info("Training CNN-BiGRU phishing detector...")

        # Extract texts and labels
        train_texts = [d['text'] for d in train_data]
        train_labels = [1 if d['label'] == 'phishing' else 0 for d in train_data]

        # Build vocabulary
        self._build_vocabulary(train_texts)

        # Convert texts to indices
        X_train = np.array([self._text_to_indices(text) for text in train_texts])
        y_train = np.array(train_labels)

        # Create PyTorch dataset
        train_dataset = torch.utils.data.TensorDataset(
            torch.LongTensor(X_train),
            torch.LongTensor(y_train)
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True
        )

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # Training loop
        num_epochs = 5  # Reduced for faster experimentation
        self.model.train()

        for epoch in range(num_epochs):
            total_loss = 0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                if batch_idx % 10 == 0:
                    logger.debug(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, "
                               f"Loss: {loss.item():.4f}")

            acc = correct / total
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, "
                       f"Accuracy={acc:.3f}")

        # Validation
        if val_data:
            self.model.eval()
            val_texts = [d['text'] for d in val_data]
            val_labels = [1 if d['label'] == 'phishing' else 0 for d in val_data]

            X_val = np.array([self._text_to_indices(text) for text in val_texts])
            y_val = np.array(val_labels)

            with torch.no_grad():
                X_val_tensor = torch.LongTensor(X_val).to(self.device)
                y_val_tensor = torch.LongTensor(y_val).to(self.device)

                output = self.model(X_val_tensor)
                _, predicted = output.max(1)
                val_acc = predicted.eq(y_val_tensor).sum().item() / len(y_val)

                logger.info(f"Validation accuracy: {val_acc:.3f}")

    def predict(self, email_dict: Dict) -> str:
        """Predict if an email is phishing"""
        text = email_dict.get('text', '')

        if self.model is None or not self.word_to_idx:
            # Fallback to simple heuristic if model not trained
            return self._fallback_predict(text)

        # Convert text to indices
        indices = self._text_to_indices(text)
        X = torch.LongTensor([indices]).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            output = self.model(X)
            _, predicted = output.max(1)

        return 'phishing' if predicted.item() == 1 else 'legitimate'

    def _fallback_predict(self, text: str) -> str:
        """Fallback prediction using simple heuristics"""
        text_lower = text.lower()

        # Key phishing indicators from the paper
        phishing_indicators = [
            'urgent', 'verify account', 'suspended', 'click here',
            'limited time', 'act now', 'confirm identity',
            'update payment', 'unusual activity', 'security alert',
            'prize', 'winner', 'congratulations', 'claim',
            'refund', 'tax', 'invoice', 'billing'
        ]

        # URL shorteners and suspicious domains
        suspicious_domains = [
            'bit.ly', 'tinyurl', 'goo.gl', 'ow.ly',
            '.tk', '.ml', '.ga', '.cf'
        ]

        score = 0

        # Check for phishing indicators
        for indicator in phishing_indicators:
            if indicator in text_lower:
                score += 1

        # Check for suspicious domains
        for domain in suspicious_domains:
            if domain in text_lower:
                score += 2

        # Check for urgency patterns
        import re
        urgency_patterns = [
            r'within \d+ (hours?|days?)',
            r'expires? (today|tomorrow|soon)',
            r'immediate(ly)?',
            r'asap'
        ]

        for pattern in urgency_patterns:
            if re.search(pattern, text_lower):
                score += 1

        # Decision based on score
        return 'phishing' if score >= 3 else 'legitimate'

    def __call__(self, email_dict: Dict) -> str:
        """Make the detector callable"""
        return self.predict(email_dict)