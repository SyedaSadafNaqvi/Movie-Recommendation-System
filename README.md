# 🎬 Movie Recommendation System

An end-to-end Machine Learning project using Collaborative Filtering to predict movie ratings and suggest Top-N recommendations.

## 🚀 Overview
This project implements a recommendation engine based on the MovieLens dataset. It compares three different algorithms and provides a production-ready inference script.

### Key Features
- **Algorithm Comparison**: User-Based CF, Item-Based CF, and Matrix Factorization (SVD).
- **Cold-Start Handling**: Implements global and movie-index averages for new users/items.
- **Top-N Recommendations**: Automatically suggests the best movies for a given user.
- **Packaged Model**: Pre-trained similarity matrices for instant predictions.

## 📁 Project Structure
- `recommendation_system.ipynb`: Full research and EDA notebook.
- `ratings.csv`: The dataset containing user interactions.
- `best_model.joblib`: The saved similarity matrix.
- `preprocessor.joblib`: The saved ID mappings and baselines.
- `inference.py`: Python script for making predictions and getting recommendations.
- `report.txt`: A simple summary of the project results.

## 🛠️ How to Run

### 1. Installation
Ensure you have the required libraries:
```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn
```

### 2. Training (Optional)
If you want to re-train the model or update the similarity matrices:
```bash
python save_artifacts.py
```

### 3. Making Predictions
To get recommendations for a user (e.g., User ID 1):
```bash
python inference.py
```

## 📊 Results
The **Item-Based Collaborative Filtering** model performed best with an **RMSE of 0.92**, proving to be both accurate and stable for this dataset.

---
*Created by Moiz Mansoori during the ML Internship.*
