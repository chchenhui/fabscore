"""
Data loader for phishing email datasets
Handles multiple open-source datasets
"""

import os
import json
import random
import pandas as pd
import numpy as np
import urllib.request
import zipfile
import csv
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class PhishingDataLoader:
    """Load and prepare phishing email datasets"""
    
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def load_and_split_data(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Load datasets and split into train/val/test"""
        
        # Load multiple datasets
        all_data = []
        
        # 1. Load built-in sample dataset
        logger.info("Loading sample phishing dataset...")
        sample_data = self._load_sample_dataset()
        all_data.extend(sample_data)
        
        # 2. Try to load CSV datasets if available
        csv_files = [
            "phishing_emails.csv",
            "spam_ham.csv",
            "email_dataset.csv"
        ]
        
        for csv_file in csv_files:
            file_path = os.path.join(self.data_dir, csv_file)
            if os.path.exists(file_path):
                logger.info(f"Loading {csv_file}...")
                data = self._load_csv_dataset(file_path)
                all_data.extend(data)
        
        # If we don't have enough data, generate synthetic samples
        if len(all_data) < 1000:
            logger.info("Generating synthetic phishing samples...")
            synthetic_data = self._generate_synthetic_samples(1000 - len(all_data))
            all_data.extend(synthetic_data)
        
        # Shuffle data
        random.shuffle(all_data)
        
        # Split into train/val/test
        n_samples = len(all_data)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        train_data = all_data[:n_train]
        val_data = all_data[n_train:n_train + n_val]
        test_data = all_data[n_train + n_val:]
        
        return train_data, val_data, test_data
    
    def _load_sample_dataset(self):
        """Load a built-in sample dataset"""
        samples = []
        
        # Phishing samples
        phishing_samples = [
            {
                "subject": "Urgent: Verify Your Account Now",
                "body": "Dear Customer, Your account has been temporarily suspended. Click here immediately to verify your identity and restore access. http://suspicious-link.com/verify",
                "sender": "security@bankk.com",
                "label": 1
            },
            {
                "subject": "You've Won $1,000,000!",
                "body": "Congratulations! You are our lucky winner. Claim your prize now by clicking this link and entering your bank details. Act fast, offer expires in 24 hours!",
                "sender": "lottery@winner-notification.net",
                "label": 1
            },
            {
                "subject": "IRS Tax Refund Notification",
                "body": "You are eligible for a tax refund of $3,458.23. Please click here to submit your bank information for direct deposit. This is time-sensitive.",
                "sender": "noreply@irs-gov.net",
                "label": 1
            },
            {
                "subject": "Account Security Alert",
                "body": "We detected unusual activity on your account from IP 192.168.1.1. If this wasn't you, click here immediately to secure your account.",
                "sender": "alert@paypal-security.org",
                "label": 1
            },
            {
                "subject": "Package Delivery Failed",
                "body": "Your package could not be delivered. Please confirm your address and provide payment for redelivery by clicking here: http://track-package.info",
                "sender": "delivery@fedex-tracking.net",
                "label": 1
            }
        ]
        
        # Legitimate samples
        legitimate_samples = [
            {
                "subject": "Meeting Reminder",
                "body": "Hi team, Just a reminder that we have our weekly sync meeting tomorrow at 2 PM. Please review the agenda beforehand. Thanks!",
                "sender": "john.doe@company.com",
                "label": 0
            },
            {
                "subject": "Project Update",
                "body": "Hello everyone, I wanted to update you on the project status. We've completed phase 1 and are moving to phase 2 next week. Let me know if you have questions.",
                "sender": "project.manager@company.com",
                "label": 0
            },
            {
                "subject": "Newsletter - March Edition",
                "body": "Check out our latest newsletter featuring industry insights, upcoming events, and team highlights. Read more on our website.",
                "sender": "newsletter@legitimate-company.com",
                "label": 0
            },
            {
                "subject": "Invoice #12345",
                "body": "Please find attached the invoice for services rendered in February. Payment is due within 30 days. Thank you for your business.",
                "sender": "accounting@vendor.com",
                "label": 0
            },
            {
                "subject": "Welcome to Our Service",
                "body": "Thank you for signing up! Your account has been created successfully. You can now log in using your registered email address.",
                "sender": "welcome@service.com",
                "label": 0
            }
        ]
        
        samples.extend(phishing_samples)
        samples.extend(legitimate_samples)
        
        # Generate variations
        for _ in range(20):
            # Create variations of phishing emails
            base = random.choice(phishing_samples)
            variation = self._create_variation(base, is_phishing=True)
            samples.append(variation)
            
            # Create variations of legitimate emails
            base = random.choice(legitimate_samples)
            variation = self._create_variation(base, is_phishing=False)
            samples.append(variation)
        
        return samples
    
    def _create_variation(self, base_email, is_phishing):
        """Create variations of emails"""
        variations = {
            "phishing_subjects": [
                "Action Required: Verify Your Account",
                "Security Alert: Suspicious Activity Detected",
                "Claim Your Reward Now",
                "Important: Update Your Payment Information",
                "Your Account Will Be Closed"
            ],
            "phishing_phrases": [
                "click here immediately",
                "verify your identity",
                "update your information",
                "confirm your account",
                "act now"
            ],
            "legitimate_subjects": [
                "Team Meeting Notes",
                "Quarterly Report",
                "Product Update",
                "Customer Feedback",
                "Schedule Change"
            ],
            "legitimate_phrases": [
                "please review",
                "for your information",
                "as discussed",
                "following up on",
                "thank you for"
            ]
        }
        
        if is_phishing:
            subject = random.choice(variations["phishing_subjects"])
            phrase = random.choice(variations["phishing_phrases"])
            body = f"{base_email['body'][:50]} {phrase} {base_email['body'][50:]}"
            sender = base_email["sender"].replace("@", str(random.randint(1, 9)) + "@")
        else:
            subject = random.choice(variations["legitimate_subjects"])
            phrase = random.choice(variations["legitimate_phrases"])
            body = f"{phrase} {base_email['body']}"
            sender = base_email["sender"]
        
        return {
            "subject": subject,
            "body": body,
            "sender": sender,
            "label": base_email["label"]
        }
    
    def _generate_synthetic_samples(self, n_samples):
        """Generate synthetic phishing and legitimate email samples"""
        samples = []
        
        phishing_templates = [
            "Your {account} needs immediate verification. Click {link} to avoid suspension.",
            "Congratulations! You've won {prize}. Claim now at {link}.",
            "Security alert: Unauthorized access to your {service}. Verify at {link}.",
            "Payment of {amount} failed. Update payment method at {link}.",
            "Your {service} subscription expires today. Renew at {link}."
        ]
        
        legitimate_templates = [
            "Reminder: {event} scheduled for {time}. Please confirm attendance.",
            "Thank you for your recent purchase. Your order {order_id} has been shipped.",
            "Monthly report for {month} is now available for review.",
            "Following up on our discussion about {topic}. Let me know your thoughts.",
            "Welcome to {service}. Here's how to get started."
        ]
        
        services = ["PayPal", "Amazon", "Netflix", "Bank", "Apple"]
        
        for i in range(n_samples):
            if i % 2 == 0:
                # Generate phishing email
                template = random.choice(phishing_templates)
                email = {
                    "subject": f"Urgent: Action Required for Your {random.choice(services)} Account",
                    "body": template.format(
                        account=random.choice(services),
                        link="http://suspicious-link-" + str(random.randint(100, 999)) + ".com",
                        prize="$" + str(random.randint(100, 10000)),
                        service=random.choice(services),
                        amount="$" + str(random.randint(10, 500))
                    ),
                    "sender": f"noreply@{random.choice(services).lower()}-security.net",
                    "label": 1
                }
            else:
                # Generate legitimate email
                template = random.choice(legitimate_templates)
                email = {
                    "subject": f"{random.choice(['Meeting', 'Update', 'Reminder', 'Follow-up'])}: {random.choice(['Project', 'Team', 'Client', 'Weekly'])}",
                    "body": template.format(
                        event="Team Meeting",
                        time="2:00 PM EST",
                        order_id="#" + str(random.randint(10000, 99999)),
                        month="March",
                        topic="project timeline",
                        service="Our Platform"
                    ),
                    "sender": f"team@legitimate-company.com",
                    "label": 0
                }
            
            samples.append(email)
        
        return samples
    
    def _load_csv_dataset(self, file_path):
        """Load dataset from CSV file"""
        samples = []
        
        try:
            df = pd.read_csv(file_path)
            
            # Try to identify columns
            text_col = None
            label_col = None
            
            for col in df.columns:
                if 'text' in col.lower() or 'body' in col.lower() or 'content' in col.lower():
                    text_col = col
                if 'label' in col.lower() or 'spam' in col.lower() or 'phish' in col.lower():
                    label_col = col
            
            if text_col and label_col:
                for _, row in df.iterrows():
                    samples.append({
                        "subject": "",
                        "body": str(row[text_col]),
                        "sender": "",
                        "label": int(row[label_col])
                    })
        except Exception as e:
            logger.warning(f"Could not load {file_path}: {e}")
        
        return samples
    
    def download_datasets(self):
        """Download public phishing datasets"""
        logger.info("Attempting to download public datasets...")
        
        # URLs for public datasets
        datasets = [
            {
                "name": "Phishing Emails",
                "url": "https://raw.githubusercontent.com/frohoff/phishing_urls_dataset/master/phishing_urls.csv",
                "file": "phishing_urls.csv"
            }
        ]
        
        for dataset in datasets:
            try:
                file_path = os.path.join(self.data_dir, dataset["file"])
                if not os.path.exists(file_path):
                    logger.info(f"Downloading {dataset['name']}...")
                    urllib.request.urlretrieve(dataset["url"], file_path)
                    logger.info(f"Downloaded {dataset['name']} successfully")
            except Exception as e:
                logger.warning(f"Could not download {dataset['name']}: {e}")