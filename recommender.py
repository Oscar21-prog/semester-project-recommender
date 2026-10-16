import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import readline  # Enables arrow keys navigation in the Ubuntu terminal

class MusicRecommender:
    def __init__(self, train_path='train.csv', test_path='test.csv'):
        print("⚙️ Loading music databases...")
        
        # Load and combine both datasets
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        self.df = pd.concat([df_train, df_test], ignore_index=True).fillna(0)
        
        # Drop duplicate tracks across datasets
        self.df = self.df.drop_duplicates(subset=['Track Name', 'Artist Name']).reset_index(drop=True)
        print(f"📦 Total tracks loaded: {len(self.df)}")

        # Normalization for search index
        self.df['search_key'] = self.df['Track Name'].astype(str).apply(lambda x: " ".join(x.lower().split()))
        
        # Feature preparation and scaling
        self.features = ['danceability', 'energy', 'acousticness', 'instrumentalness', 'valence', 'tempo']
        self.scaler = StandardScaler()
        
        matrix = self.scaler.fit_transform(self.df[self.features])
        self.similarities = cosine_similarity(matrix)
        print("✅ Recommendation engine initialized and ready!\n")

    def _find_song_index(self, clean_input):
        """Internal helper to resolve exact, partial, or typo-tolerant song matches."""
        # 1. Exact match
        exact_matches = self.df[self.df['search_key'] == clean_input]
        if not exact_matches.empty:
            return exact_matches.sort_values(by='Popularity', ascending=False).index[0]

        # 2. Substring match (e.g. 'bohemian rhapsody' in 'bohemian rhapsody - remastered 2011')
        contains_matches = self.df[self.df['search_key'].str.contains(clean_input, regex=False)]
        if not contains_matches.empty:
            return contains_matches.sort_values(by='Popularity', ascending=False).index[0]

        # 3. Strict typo matching (cutoff raised to 0.82 + keyword presence check)
        all_keys = self.df['search_key'].tolist()
        closest = difflib.get_close_matches(clean_input, all_keys, n=1, cutoff=0.82)
        
        if closest:
            best_key = closest[0]
            # Extra verification: ensure major words (> 2 chars) exist in the matched title
            input_words = [w for w in clean_input.split() if len(w) > 2]
            best_key_words = set(best_key.split())
            
            # Check if at least the longest/main word appears in the match
            if any(word in best_key_words for word in input_words):
                return self.df[self.df['search_key'] == best_key].sort_values(by='Popularity', ascending=False).index[0]

        return None

    def recommend(self, song_name, top_n=5):
        """Public method to get recommendations with smart resolution and duplicate filtering."""
        clean_input = " ".join(song_name.lower().split())
        
        idx = self._find_song_index(clean_input)
        
        if idx is None:
            print(f"❌ Error: '{song_name}' is not in the database (or no close match found).")
            return

        original_name = self.df.iloc[idx]['Track Name']
        original_artist = self.df.iloc[idx]['Artist Name']
        
        print(f"\n🎯 Selected track: '{original_name}' by {original_artist}")
        print(f"🎧 Because you listened to this, we recommend:")

        scores = list(enumerate(self.similarities[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        seen_songs = set()
        recommended_count = 0

        for i in range(1, len(scores)):
            song_idx = scores[i][0]
            score = scores[i][1]
            artist = self.df.iloc[song_idx]['Artist Name']
            song = self.df.iloc[song_idx]['Track Name']

            signature = f"{song} - {artist}"

            if signature not in seen_songs:
                seen_songs.add(signature)
                match_percentage = round(score * 100, 1)
                print(f"  - {song} (by {artist}) | {match_percentage}% match")
                recommended_count += 1

            if recommended_count == top_n:
                break


if __name__ == "__main__":
    my_recommender = MusicRecommender('train.csv', 'test.csv')

    while True:
        user_input = input("\nEnter a song name (or 'exit' to quit): ")

        if user_input.strip().lower() == 'exit':
            print("Goodbye!")
            break

        if not user_input.strip():
            continue

        my_recommender.recommend(user_input)