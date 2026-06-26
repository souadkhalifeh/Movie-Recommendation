# 🎬 Movie Recommender

A hands-on movie recommendation project built on the [MovieLens](https://grouplens.org/datasets/movielens/) dataset. It implements and compares three classic recommendation approaches in a single notebook, from raw ratings all the way to personalised suggestions.

---

## 📦 Dataset

The project uses the MovieLens small dataset (`./data`):

| File | Shape | Contents |
|------|-------|----------|
| `movies.csv` | 9,742 × 3 | `movieId`, `title`, `genres` (pipe-separated) |
| `ratings.csv` | 100,836 × 4 | `userId`, `movieId`, `rating`, `timestamp` |
| `tags.csv` | 3,683 × 4 | `userId`, `movieId`, `tag`, `timestamp` |

---

## 🧠 Methods

### 1. Item-Based Collaborative Filtering (KNN)

Recommends movies based on **rating patterns** — "people who rated this movie the way you did also rated these."

- Built a **movie × user** rating matrix by pivoting `ratings` (`movieId` as rows, `userId` as columns) and filling missing ratings with `0`.
- **Filtered out noise** to keep the signal strong:
  - movies with **more than 10** user votes,
  - users who rated **more than 50** movies.
  - This shrank the matrix from `(9742 × 610)` down to `(2121 × 378)`.
- Converted the dense matrix to a **sparse `csr_matrix`** for memory efficiency (the rating matrix is very sparse).
- Trained a **`NearestNeighbors`** model with:
  - `metric='cosine'` (cosine distance between movie rating vectors),
  - `algorithm='brute'`,
  - `n_neighbors=20`.
- `get_movie_recommendation(title)` looks up a movie by (case-insensitive) title, queries the KNN model for its nearest neighbours, drops the movie itself, and returns the closest titles with their cosine distance.

**Strengths:** captures real user taste / "movies watched together."
**Weakness:** cold-start — a movie with too few ratings can't be recommended well.

### 2. Content-Based Filtering (TF-IDF)

Recommends movies that are **described** similarly, using their **genres + title** as text — no ratings required, so no cold-start for new movies.

- Built a text "document" per movie: replaced `|` in genres with spaces and appended the lower-cased title (e.g. `"Adventure Animation Children Comedy Fantasy toy story (1995)"`).
- Vectorised with **`TfidfVectorizer`**:
  - `stop_words='english'`,
  - `ngram_range=(1, 2)` to capture two-word phrases like *"science fiction"*,
  - `min_df=2` to drop ultra-rare terms.
  - Result: a `(9742 × 5701)` TF-IDF matrix.
- Computed **cosine similarity one query at a time** instead of building the full `9742 × 9742` similarity matrix (which would be ~720 MB of dense floats and could crash a laptop). For a given movie we compare only *its* vector against all others — a cheap `(1 × n)` operation.
- `get_similar_movies(movie_id, n)` returns the top-N most similar movies with their similarity scores.

**Strengths:** works for new/rarely-rated movies, fully explainable.
**Weakness:** limited to what's literally written in the metadata.

### 3. Personalised Content-Based Recommendations

Turns the content model into a **per-user** recommender.

- `user_recommendations(user_id, n)` builds a **user profile** by averaging the similarity vectors of every movie the user rated, **weighted by the rating they gave**.
- Already-watched movies are masked out (`-inf`) so they're never recommended.
- Returns the highest-scoring **unwatched** movies for that user.

### 4. Comparison

The final section picks a movie present in both worlds (e.g. *Toy Story*) and shows the collaborative vs. content-based recommendations **side by side**, highlighting how each method "thinks" differently — ratings-based co-watching vs. metadata similarity.

---

## 🛠️ Tech Stack

- **Python**
- **pandas** / **numpy** — data wrangling
- **scipy** — sparse matrices (`csr_matrix`)
- **scikit-learn** — `NearestNeighbors`, `TfidfVectorizer`, `cosine_similarity`
- **matplotlib** — exploratory plots (votes per movie / per user)

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone <repo-url>
cd AI_BUILDERS_Movie_Recommender

# 2. (Optional) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install pandas numpy scipy scikit-learn matplotlib jupyter

# 4. Launch the notebook
jupyter notebook movie_recomm.ipynb
```

Make sure `movies.csv`, `ratings.csv`, and `tags.csv` are in the `./data` folder.

---

## 📊 Example Usage

```python
# Collaborative filtering
get_movie_recommendation('Iron Man')

# Content-based
get_similar_movies(1, 10)        # similar to Toy Story (movieId = 1)

# Personalised
user_recommendations(173, 10)    # top 10 for user 173
```

---

## 📁 Project Structure

```
AI_BUILDERS_Movie_Recommender/
├── data/
│   ├── movies.csv
│   ├── ratings.csv
│   └── tags.csv
├── movie_recomm.ipynb     # main notebook (all methods)
├── .gitignore
└── README.md
```
