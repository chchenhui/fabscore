"""
BERT-based Phishing Email Detector
Based on recent research (2023-2024) using transformer models for phishing detection
References:
- "BERT-Phish: Pre-training BERT for Phishing Detection" (2023)
- "Deep Learning-based Phishing Email Detection Using Transformers" (2024)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class BERTPhishingDetector:
    """BERT-based detector using distilled models for efficiency"""

    def __init__(self, model_name="distilbert", max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.classifier = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Try to import transformers
        try:
            from transformers import AutoTokenizer, AutoModel
            self.AutoTokenizer = AutoTokenizer
            self.AutoModel = AutoModel
            self._initialize_bert()
        except ImportError:
            logger.warning("Transformers library not available. Using fallback method.")
            self._use_fallback()

    def _initialize_bert(self):
        """Initialize BERT model and tokenizer"""
        try:
            # Use smaller distilbert for efficiency
            self.tokenizer = self.AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = self.AutoModel.from_pretrained("distilbert-base-uncased")
            self.model.to(self.device)
            self.model.eval()

            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 2)
            ).to(self.device)

            logger.info("BERT model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize BERT: {e}")
            self._use_fallback()

    def _use_fallback(self):
        """Fallback to simpler embedding-based approach"""
        logger.info("Using fallback embedding-based approach")
        self.use_fallback = True

        # Simple word embedding approach
        self.vocab_size = 10000
        self.embedding_dim = 128
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, 64, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def extract_features(self, texts: List[str]) -> np.ndarray:
        """Extract BERT features from texts"""
        if hasattr(self, 'use_fallback') and self.use_fallback:
            return self._extract_fallback_features(texts)

        features = []

        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = self.tokenizer(
                    text[:self.max_length * 3],  # Rough character limit
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                ).to(self.device)

                # Get BERT embeddings
                outputs = self.model(**inputs)

                # Use [CLS] token representation
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                features.append(cls_embedding[0])

        return np.array(features)

    def _extract_fallback_features(self, texts: List[str]) -> np.ndarray:
        """Extract features using fallback method"""
        features = []

        for text in texts:
            # Simple tokenization
            words = text.lower().split()[:self.max_length]

            # Convert to indices (simple hash)
            indices = [hash(word) % self.vocab_size for word in words]

            # Create feature vector
            feature = np.zeros(128)

            # Aggregate word features
            for idx in indices:
                embed = self.embeddings(torch.tensor([idx]))
                feature += embed.detach().numpy()[0]

            if len(indices) > 0:
                feature /= len(indices)

            features.append(feature)

        return np.array(features)

    def train(self, train_data: List[Dict], val_data: List[Dict] = None):
        """Train the BERT-based classifier"""
        logger.info("Training BERT-based phishing detector...")

        # Extract texts and labels
        train_texts = [d['text'] for d in train_data]
        train_labels = [1 if d['label'] == 'phishing' else 0 for d in train_data]

        # Extract features
        X_train = self.extract_features(train_texts)
        y_train = np.array(train_labels)

        # Train a simple classifier on top of BERT features
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        self.sklearn_classifier = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        self.sklearn_classifier.fit(X_train_scaled, y_train)

        # Validation
        if val_data:
            val_texts = [d['text'] for d in val_data]
            val_labels = [1 if d['label'] == 'phishing' else 0 for d in val_data]
            X_val = self.extract_features(val_texts)
            X_val_scaled = self.scaler.transform(X_val)

            val_acc = self.sklearn_classifier.score(X_val_scaled, val_labels)
            logger.info(f"Validation accuracy: {val_acc:.3f}")

    def predict(self, email_dict: Dict) -> str:
        """Predict if an email is phishing"""
        text = email_dict.get('text', '')

        # Extract features
        features = self.extract_features([text])

        if hasattr(self, 'scaler'):
            features = self.scaler.transform(features)

        # Predict
        if hasattr(self, 'sklearn_classifier'):
            prediction = self.sklearn_classifier.predict(features)[0]
            return 'phishing' if prediction == 1 else 'legitimate'
        else:
            # Fallback: simple heuristic
            phishing_keywords = ['urgent', 'verify', 'suspended', 'click here',
                                'limited time', 'act now', 'confirm identity']
            text_lower = text.lower()
            score = sum(1 for keyword in phishing_keywords if keyword in text_lower)
            return 'phishing' if score >= 2 else 'legitimate'

    def __call__(self, email_dict: Dict) -> str:
        """Make the detector callable"""
        return self.predict(email_dict)