import difflib
from pathlib import Path
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

try:
    import readline  # Improves arrow-key navigation in Unix terminals
except ImportError:
    readline = None


def resolve_data_path(filename: str) -> str:
    """Locate a dataset file across standard project directory structures."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "data" / filename,
        script_dir / filename,
        Path.cwd() / "data" / filename,
        Path.cwd() / filename,
        Path.cwd() / "src" / "data" / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return filename


class MusicRecommender:
    def __init__(self, train_path: str = None, test_path: str = None):
        print("⚙️ Loading music databases...")

        train_file = train_path or resolve_data_path("train.csv")
        test_file = test_path or resolve_data_path("test.csv")

        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        self.df = pd.concat([train_df, test_df], ignore_index=True)

        required_columns = {
            "Track Name",
            "Artist Name",
            "Popularity",
            "danceability",
            "energy",
            "acousticness",
            "instrumentalness",
            "valence",
            "tempo",
        }
        missing = required_columns - set(self.df.columns)
        if missing:
            raise ValueError(
                "The following columns are missing from the CSV files: "
                + ", ".join(sorted(missing))
            )

        # Deduplicate identical song entries across both splits
        self.df = self.df.drop_duplicates(
            subset=["Track Name", "Artist Name"]
        ).reset_index(drop=True)

        # Build normalized keys for robust query matching
        self.df["search_key"] = (
            self.df["Track Name"]
            .fillna("")
            .astype(str)
            .apply(lambda name: " ".join(name.casefold().split()))
        )
        self.df["artist_key"] = (
            self.df["Artist Name"]
            .fillna("")
            .astype(str)
            .apply(lambda name: " ".join(name.casefold().split()))
        )

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
        )
        feature_data = feature_data.fillna(feature_data.median()).fillna(0)

        # Standardize features (mean=0, variance=1)
        self.scaler = StandardScaler()
        feature_matrix = self.scaler.fit_transform(feature_data)

        # FEATURE WEIGHTING: Assign custom weights to increase accuracy
        # Energy and Tempo have higher influence on musical style compatibility.
        self.feature_weights = {
            "danceability": 1.2,
            "energy": 1.5,          # High weight: intensity matters
            "acousticness": 1.0,
            "instrumentalness": 0.8,
            "valence": 1.1,         # Mood influence
            "tempo": 1.4            # High weight: BPM alignment matters
        }

        # Apply weights across feature columns
        for idx, feature in enumerate(self.features):
            weight = self.feature_weights.get(feature, 1.0)
            feature_matrix[:, idx] *= weight

        self.feature_matrix = feature_matrix

        print(f"📦 Total unique tracks indexed: {len(self.df)}")
        print("✅ Recommendation engine initialized and ready (with custom feature weighting)!\n")

    def _most_popular_index(self, indexes):
        """Select the highest-popularity track index among multiple candidates."""
        if len(indexes) == 1:
            return indexes[0]

        popularity = pd.to_numeric(
            self.df.loc[indexes, "Popularity"], errors="coerce"
        ).fillna(-1)
        return popularity.idxmax()

    def _find_song_index(self, song_name: str):
        """Resolve track queries using exact, partial substring, and fuzzy matching."""
        query = " ".join(song_name.casefold().split())

        if len(query) < 2:
            return None, None

        # 1. Exact match
        exact = self.df.index[self.df["search_key"] == query].tolist()
        if exact:
            return self._most_popular_index(exact), "exact match"

        # 2. Substring match
        partial = self.df.index[
            self.df["search_key"].str.contains(query, regex=False, na=False)
        ].tolist()

        if partial:
            if len(partial) == 1:
                return partial[0], "partial match"

            print(f"\nMultiple tracks match '{song_name}':")
            display_limit = min(len(partial), 10)
            for option_number, track_index in enumerate(
                partial[:display_limit], start=1
            ):
                row = self.df.iloc[track_index]
                print(
                    f"  {option_number}. {row['Track Name']} — {row['Artist Name']}"
                )

            if len(partial) > 10:
                print(f"  ... and {len(partial) - 10} more matches.")

            choice = input(
                "\nEnter track number to select (or press Enter to cancel): "
            ).strip()

            if not choice:
                return None, None

            if not choice.isdigit() or not 1 <= int(choice) <= display_limit:
                print("Invalid selection.")
                return None, None

            return partial[int(choice) - 1], "partial match"

        # 3. Fuzzy matching with keyword verification
        unique_keys = self.df["search_key"].drop_duplicates().tolist()
        matches = difflib.get_close_matches(query, unique_keys, n=1, cutoff=0.80)

        if not matches:
            return None, None

        candidate = matches[0]
        query_words = {word for word in query.split() if len(word) > 2}
        candidate_words = set(candidate.split())

        if query_words and not query_words.intersection(candidate_words):
            return None, None

        indexes = self.df.index[self.df["search_key"] == candidate].tolist()
        return self._most_popular_index(indexes), "approximate match"

    def search_artist(self, artist_name: str, limit: int = 15):
        """List catalog tracks belonging to a given artist query."""
        query = " ".join(artist_name.casefold().split())
        if not query:
            print("Please provide a valid artist name.")
            return

        matches = self.df[
            self.df["artist_key"].str.contains(query, regex=False, na=False)
        ]

        if matches.empty:
            print(f"❌ No tracks found for artist '{artist_name}'.")
            return

        if "Popularity" in matches.columns:
            matches = matches.assign(
                _pop=pd.to_numeric(matches["Popularity"], errors="coerce").fillna(-1)
            ).sort_values("_pop", ascending=False)

        print(f"\n🎤 Available tracks for '{artist_name}':")
        for _, row in matches.head(limit).iterrows():
            print(f"  - {row['Track Name']} (Popularity: {row['Popularity']})")

        if len(matches) > limit:
            print(f"  ... and {len(matches) - limit} additional tracks.")

    def recommend(self, song_name: str, top_n: int = 5):
        """Compute cosine similarity on demand and return top recommendations."""
        index, match_type = self._find_song_index(song_name)

        if index is None:
            print(f"❌ Track '{song_name}' not found in database.")
            print("Tip: Use Option 2 to check existing tracks by artist.")
            return

        selected = self.df.iloc[index]
        print(
            f"\n🎯 Selected track: '{selected['Track Name']}' by {selected['Artist Name']}"
        )
        print(f"   Match resolution: {match_type}")
        print("🎧 Recommendations based on weighted acoustic similarity:")

        # Compute cosine similarity on-the-fly against the weighted feature matrix
        target_vector = self.feature_matrix[index].reshape(1, -1)
        similarity_scores = cosine_similarity(target_vector, self.feature_matrix)[0]

        # Rank all indices in descending order of similarity
        ranked_indices = similarity_scores.argsort()[::-1]

        seen_signatures = set()
        delivered = 0

        for candidate_idx in ranked_indices:
            if candidate_idx == index:
                continue

            row = self.df.iloc[candidate_idx]
            signature = (
                str(row["Track Name"]).casefold(),
                str(row["Artist Name"]).casefold(),
            )

            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

            score = similarity_scores[candidate_idx]
            match_pct = round(score * 100, 1)

            print(
                f"  {delivered + 1}. {row['Track Name']} — {row['Artist Name']} "
                f"| {match_pct}% match (cosine: {score:.3f})"
            )

            delivered += 1
            if delivered >= top_n:
                break


if __name__ == "__main__":
    recommender = MusicRecommender()

    while True:
        print("\n" + "=" * 45)
        print("MUSIC RECOMMENDER SYSTEM (CLI PROTOTYPE)")
        print("=" * 45)
        print("1. Get recommendations from track title")
        print("2. Search available tracks by artist")
        print("3. Exit")

        choice = input("\nSelect an option (1-3): ").strip()

        if choice == "1":
            title = input("Enter track title: ").strip()
            if title:
                recommender.recommend(title)
        elif choice == "2":
            artist = input("Enter artist name: ").strip()
            if artist:
                recommender.search_artist(artist)
        elif choice == "3":
            print("Exiting application. Goodbye!")
            break
        else:
            print("Invalid selection. Please choose 1, 2, or 3.")