# Hybrid Movie Recommendation System

A movie recommender built on the MovieLens dataset. It combines two approaches:

- **Collaborative filtering:** a matrix-factorization model (SVD from `scikit-surprise`) learns user and movie factors from ratings and predicts how much a user would rate an unseen movie.
- **Content-based filtering:** TF-IDF vectors over movie genres, compared with cosine similarity, find movies similar to one the user liked.
- **Hybrid:** the content-based candidates for a seed movie are re-ranked by the SVD-predicted rating for that specific user, so two users who like the same film get different lists.

## Results

| Model | Split | Metric | Score |
|---|---|---|---|
| SVD (collaborative filtering) | 75% train / 25% test | RMSE | **0.8828** |

RMSE is on the 0.5–5 star rating scale. The split is random with no fixed seed, so re-runs vary slightly.

## How it works

1. Load `movies.csv` (movieId, title, genres) and `ratings.csv` (userId, movieId, rating, timestamp), merge and clean them.
2. Train SVD on the ratings with `surprise` and evaluate RMSE on the held-out 25%.
3. Build a TF-IDF matrix over the genre strings and compute pairwise cosine similarity.
4. `recommend_movies(title)` returns the movies with the most similar genres.
5. `hybrid_recommendation(user_id, title)` takes those candidates and sorts them by the SVD-predicted rating for that user.

## Example

```python
recommend_movies("Toy Story (1995)")
# Antz (1998), Toy Story 2 (1999), The Emperor's New Groove (2000), ...

hybrid_recommendation(user_id=1, movie_title="Toy Story (1995)")
# ['Antz (1998)', 'Monsters, Inc. (2001)', 'Toy Story 2 (1999)', ...]
```

## Run it

```bash
pip install pandas scikit-surprise scikit-learn matplotlib jupyter
jupyter notebook Movie_Recommendation_System_Clean.ipynb
```

**Data:**

- The movie file in this repo is `movie.csv`; the notebook reads `movies.csv`. Rename the file, or change the path in the first cell.
- `ratings.csv` is too large for GitHub. Download it from [Google Drive](https://drive.google.com/file/d/1V19rDoA7ZizQ-PJ5BMcbiilL-V6hB_gR/view?usp=sharing) and put it in the project root.

## Repository structure

```
Movie-recommendation-system/
├── Movie_Recommendation_System_Clean.ipynb   # full pipeline: loading, SVD, TF-IDF, hybrid
├── movie.csv                                  # movie metadata
└── README.md
```

## Tech stack

Python · pandas · scikit-surprise (SVD) · scikit-learn (TF-IDF, cosine similarity) · Matplotlib · Jupyter

## Next steps

- Fix a known bug: the hybrid step passes the DataFrame row index to `model.predict` instead of the `movieId`, so its re-ranking is not yet using the right predictions.
- Fix the random seed and report RMSE/MAE averaged over 5-fold cross-validation.
- Add ranking metrics (precision@k, recall@k) for the hybrid recommender.
- Use tags and plot text alongside genres for richer content features.
- Serve recommendations through a small FastAPI endpoint.

## Author

Vishal Yadav · [LinkedIn](https://linkedin.com/in/vishal-yadav-35b027281/)

# Movie Recommendation System 🎬

This project builds a **hybrid movie recommender system** using both **collaborative filtering** (SVD) and **content-based filtering** (TF-IDF on genres) on the popular MovieLens dataset.

## 📂 Dataset

- `movies.csv` — Movie metadata (movieId, title, genres)
- `ratings.csv` — User ratings (userId, movieId, rating, timestamp)

## 📌 Features

- Merge movie and rating data
- Data cleaning and preprocessing
- Collaborative filtering using **Surprise SVD**
- Content-based recommendations using **TF-IDF + Cosine Similarity**
- **Hybrid recommender** combining both approaches
- Visualized top movie recommendations with **Matplotlib**

## 🧠 Model Used

- **Collaborative Filtering**: `SVD` from `scikit-surprise`
- **Content-Based Filtering**: TF-IDF on genres with cosine similarity
- **Hybrid**: Top content-based recommendations re-ranked using predicted user ratings

## 📈 Evaluation

- Metric: **Root Mean Squared Error (RMSE)**
- Sample RMSE: ~0.88 on test data

## 🧪 How to Use

1. Install dependencies:
   ```bash
   pip install pandas scikit-surprise scikit-learn matplotlib
   ```

2. Run the notebook or Python script to:
   - Train the model
   - Test recommendation functions

3. Example usage:
   ```python
   recommend_movies("Toy Story (1995)")
   hybrid_recommendation(user_id=1, movie_title="Toy Story (1995)")
   ```

## 📊 Sample Output

```text
Recommended Movies:
- Antz (1998)
- Toy Story 2 (1999)
- The Emperor's New Groove (2000)
...

Hybrid Recommendations for User 1:
['Antz (1998)', 'Monsters, Inc. (2001)', 'Toy Story 2 (1999)', "Emperor's New Groove, The (2000)", 'Adventures of Rocky and Bullwinkle, The (2000)']
```

## 📌 Visualization

A horizontal bar chart shows top recommended movies with their respective recommendation points.

## 📁 Repository Structure

```
📦Movie-recommendation-system
 ┣ 📄 movies.csv
 ┣ 📄 ratings.csv
 ┣ 📄 Movie_Recommendation_System_Clean.ipynb
 ┣ 📄 README.md
```

## 📂 Large Dataset

The dataset `ratings.csv` is too large to upload directly to GitHub.

you can manually download it from:  
[Google Drive Link](https://drive.google.com/file/d/1V19rDoA7ZizQ-PJ5BMcbiilL-V6hB_gR/view?usp=sharing)

After downloading, place the `ratings.csv` file in the project root folder.


## ✍️ Author

**Vishal Yadav**

---

Feel free to ⭐ star the repository if you find it helpful!
