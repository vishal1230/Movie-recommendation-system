
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
