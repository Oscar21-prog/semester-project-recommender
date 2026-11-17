from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "danceability",
    "energy",
    "acousticness",
    "instrumentalness",
    "valence",
    "tempo",
]

WEIGHTS = np.array([1.2, 1.5, 1.0, 0.8, 1.1, 1.4])


def resolve_data_path(filename: str) -> str:
    """Locate a dataset file in common project directory structures."""
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
        print("Loading datasets for evaluation...")

        train_file = train_path or resolve_data_path("train.csv")
        test_file = test_path or resolve_data_path("test.csv")

        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        self.df = pd.concat([train_df, test_df], ignore_index=True)

        required_columns = {
            "Track Name",
            "Artist Name",
            "Class",
            *FEATURES,
        }
        missing_columns = required_columns - set(self.df.columns)

        if missing_columns:
            raise ValueError(
                "Missing required CSV columns: "
                + ", ".join(sorted(missing_columns))
            )

        # Match the track deduplication used by recommender.py
        self.df = self.df.drop_duplicates(
            subset=["Track Name", "Artist Name"]
        ).reset_index(drop=True)

        # Match the recommender's numeric conversion and median imputation
        feature_data = self.df[FEATURES].apply(
            pd.to_numeric, errors="coerce"
        )
        feature_data = feature_data.fillna(feature_data.median()).fillna(0)

        scaler = StandardScaler()
        self.unweighted_matrix = scaler.fit_transform(feature_data)

        # Compare the unweighted baseline with the recommender's weighted model
        self.weighted_matrix = self.unweighted_matrix * WEIGHTS

        print(f"Tracks available for evaluation: {len(self.df)}")
        print("Evaluation data prepared.\n")

    def _precision_at_k(self, matrix, query_indices, k: int) -> float:
        """Calculate mean Precision@K for a set of query tracks."""
        effective_k = min(k, len(self.df) - 1)
        if effective_k < 1:
            raise ValueError("At least two tracks are required for evaluation.")

        labels = self.df["Class"].to_numpy()
        tracks = self.df["Track Name"].fillna("").astype(str).to_numpy()
        artists = self.df["Artist Name"].fillna("").astype(str).to_numpy()

        precision_scores = []

        for query_index in query_indices:
            similarities = cosine_similarity(
                matrix[query_index].reshape(1, -1),
                matrix,
            )[0]

            ranked_indices = similarities.argsort()[::-1]
            query_signature = (
                tracks[query_index].casefold(),
                artists[query_index].casefold(),
            )

            top_k = []
            seen_tracks = set()

            for candidate_index in ranked_indices:
                if candidate_index == query_index:
                    continue

                signature = (
                    tracks[candidate_index].casefold(),
                    artists[candidate_index].casefold(),
                )

                if signature == query_signature or signature in seen_tracks:
                    continue

                seen_tracks.add(signature)
                top_k.append(candidate_index)

                if len(top_k) == effective_k:
                    break

            target_class = labels[query_index]
            hits = sum(
                1
                for candidate_index in top_k
                if pd.notna(labels[candidate_index])
                and labels[candidate_index] == target_class
            )

            precision_scores.append(hits / effective_k)

        return float(np.mean(precision_scores))

    def evaluate_precision_at_k(
        self,
        sample_size: int = 300,
        k: int = 5,
        random_state: int = 42,
    ):
        """Compare genre/class consistency for weighted and unweighted features."""
        if sample_size < 1:
            raise ValueError("sample_size must be at least 1.")
        if k < 1:
            raise ValueError("k must be at least 1.")

        valid_query_indices = np.flatnonzero(
            self.df["Class"].notna().to_numpy()
        )

        if len(valid_query_indices) == 0:
            raise ValueError(
                "The 'Class' column has no labeled tracks to evaluate."
            )

        rng = np.random.default_rng(random_state)
        sample_count = min(sample_size, len(valid_query_indices))
        query_indices = rng.choice(
            valid_query_indices,
            size=sample_count,
            replace=False,
        )

        print(
            f"Evaluating {sample_count} query tracks "
            f"with Precision@{k}..."
        )

        baseline_score = self._precision_at_k(
            self.unweighted_matrix,
            query_indices,
            k,
        )
        weighted_score = self._precision_at_k(
            self.weighted_matrix,
            query_indices,
            k,
        )

        baseline_percent = baseline_score * 100
        weighted_percent = weighted_score * 100
        difference = weighted_percent - baseline_percent

        print("\nEXPERIMENTAL RESULTS")
        print("----------------------------------------")
        print(f"Evaluated query tracks: {sample_count}")
        print(f"K: {min(k, len(self.df) - 1)}")
        print(
            "Unweighted baseline class consistency "
            f"(Precision@K): {baseline_percent:.2f}%"
        )
        print(
            "Weighted model class consistency "
            f"(Precision@K): {weighted_percent:.2f}%"
        )
        print(
            f"Difference: {difference:+.2f} percentage points"
        )
        print("----------------------------------------")

        return {
            "sample_size": sample_count,
            "k": min(k, len(self.df) - 1),
            "unweighted_precision_at_k": baseline_score,
            "weighted_precision_at_k": weighted_score,
            "difference_percentage_points": difference,
        }


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.evaluate_precision_at_k(sample_size=300, k=5)