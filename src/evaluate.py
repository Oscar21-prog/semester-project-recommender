from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


def resolve_data_path(filename: str) -> str:
    """Locate a dataset file across standard project directory structures."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "data" / filename,
        script_dir / filename,
        script_dir.parent / "data" / filename,
        script_dir.parent / filename,
        Path.cwd() / "data" / filename,
        Path.cwd() / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return filename


class ModelEvaluator:
    def __init__(self, train_path: str = None, test_path: str = None):
        print("⚙️ Loading datasets for evaluation...")

        train_file = train_path or resolve_data_path("train.csv")
        test_file = test_path or resolve_data_path("test.csv")

        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        self.df = pd.concat([train_df, test_df], ignore_index=True).fillna(0)
        self.df = self.df.drop_duplicates(
            subset=["Track Name", "Artist Name"]
        ).reset_index(drop=True)

        self.features = [
            "danceability",
            "energy",
            "acousticness",
            "instrumentalness",
            "valence",
            "tempo",
        ]

        feature_data = self.df[self.features].apply(
            pd.to_numeric, errors="coerce"
        ).fillna(0)

        scaler = StandardScaler()
        matrix = scaler.fit_transform(feature_data)

        # Apply the exact same feature weighting used in recommender.py
        weights = [1.2, 1.5, 1.0, 0.8, 1.1, 1.4]
        weighted_matrix = matrix * weights

        print("Computing similarity matrix for evaluation...")
        self.similarities = cosine_similarity(weighted_matrix)
        print("✅ Evaluation engine ready!\n")

    def evaluate_precision_at_k(self, sample_size=300, k=5):
        """Evaluate genre consistency across a random sample of tracks."""
        if "Class" not in self.df.columns:
            print("❌ Error: 'Class' column not found in dataset for evaluation.")
            return

        np.random.seed(42)
        sample_indices = np.random.choice(
            len(self.df), size=min(sample_size, len(self.df)), replace=False
        )

        hits = 0
        total = 0

        print(
            f"🔍 Evaluating Precision@{k} over {len(sample_indices)} random tracks..."
        )

        for idx in sample_indices:
            target_class = self.df.iloc[idx]["Class"]
            scores = self.similarities[idx]
            ranked_indices = scores.argsort()[::-1]

            top_k = []
            seen = set()

            for other_idx in ranked_indices:
                if other_idx == idx:
                    continue
                row = self.df.iloc[other_idx]
                signature = (
                    str(row["Track Name"]).casefold(),
                    str(row["Artist Name"]).casefold(),
                )
                if signature in seen:
                    continue
                seen.add(signature)
                top_k.append(other_idx)
                if len(top_k) == k:
                    break

            same_class_hits = sum(
                1 for i in top_k if self.df.iloc[i]["Class"] == target_class
            )
            hits += same_class_hits
            total += k

        accuracy = (hits / total) * 100
        print(f"\n📊 EXPERIMENTAL RESULTS:")
        print(f"----------------------------------------")
        print(f"Total Evaluated Queries: {len(sample_indices)}")
        print(f"Evaluated Recommendations (K={k}): {total}")
        print(f"Genre Consistency Accuracy (Precision@{k}): {accuracy:.2f}%")
        print(f"----------------------------------------")
        return accuracy


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.evaluate_precision_at_k(sample_size=300, k=5)