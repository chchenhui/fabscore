import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class SyntheticFairnessDataset:
    def __init__(self, n_samples=1000, bias_strength=0.2, random_state=42):
        np.random.seed(random_state)
        self.n_samples = n_samples
        self.bias_strength = bias_strength
        self.random_state = random_state

    def generate(self):
        np.random.seed(self.random_state)

        # Features
        age = np.random.normal(35, 10, self.n_samples)
        education = np.random.randint(0, 3, self.n_samples)  # 0=low,1=mid,2=high
        income = np.random.normal(50000, 15000, self.n_samples)

        # Protected attribute (e.g., group membership)
        group = np.random.binomial(1, 0.5, self.n_samples)

        # Base label probability using realistic coefficients
        logits = (
            0.02 * (age - 35) +  # Age effect
            0.5 * (education == 2) + 0.2 * (education == 1) +  # Education effect
            0.00001 * (income - 50000) +  # Income effect
            np.random.normal(0, 0.1, self.n_samples)  # Noise
        )

        # Inject bias: group 0 has artificially reduced chance of positive label
        logits[group == 0] -= self.bias_strength

        probs = 1 / (1 + np.exp(-logits))
        labels = np.random.binomial(1, probs)

        df = pd.DataFrame({
            "age": age,
            "education": education,
            "income": income,
            "group": group,
            "label": labels
        })
        return df

    def get_train_test_split(self, test_size=0.2):
        df = self.generate()

        # Features and target
        X = df[['age', 'education', 'income']].values
        y = df['label'].values
        a = df['group'].values

        # Train-test split
        X_train, X_test, y_train, y_test, a_train, a_test = train_test_split(
            X, y, a, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test, a_train, a_test, scaler

    def compute_bias_statistics(self):
        df = self.generate()

        # Overall statistics
        overall_positive_rate = df['label'].mean()

        # Group-wise statistics
        group_stats = df.groupby('group')['label'].agg(['mean', 'count']).reset_index()
        group_stats.columns = ['group', 'positive_rate', 'count']

        # Bias metrics
        bias_difference = (
            group_stats[group_stats['group'] == 1]['positive_rate'].iloc[0] -
            group_stats[group_stats['group'] == 0]['positive_rate'].iloc[0]
        )

        return {
            'overall_positive_rate': overall_positive_rate,
            'group_statistics': group_stats,
            'bias_difference': bias_difference,
            'bias_strength': self.bias_strength
        }

if __name__ == "__main__":
    dataset = SyntheticFairnessDataset(n_samples=1000, bias_strength=0.3)
    df = dataset.generate()
    print("Dataset sample:")
    print(df.head())
    print("\nDataset statistics:")
    print(df.describe())
    print("\nBias analysis:")
    bias_stats = dataset.compute_bias_statistics()
    print(f"Overall positive rate: {bias_stats['overall_positive_rate']:.3f}")
    print(f"Bias difference: {bias_stats['bias_difference']:.3f}")
    print(bias_stats['group_statistics'])