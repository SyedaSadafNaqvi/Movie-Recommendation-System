import joblib
import numpy as np
import pandas as pd  
def load_recommender():
    """Loads the preprocessor and model components."""
    print("Loading model components...")
    preprocessor = joblib.load('preprocessor.joblib')
    model = joblib.load('best_model.joblib')

    # If movie_averages is missing, compute it from ratings.csv
    if 'movie_averages' not in preprocessor:
        print("movie_averages not found in preprocessor. Computing from ratings.csv ...")
        ratings = pd.read_csv('ratings.csv')  # must have columns: movieId, rating
        movie_averages = ratings.groupby('movieId')['rating'].mean().to_dict()
        preprocessor['movie_averages'] = movie_averages

        # Optional: save back to disk so next time it's already there
        joblib.dump(preprocessor, 'preprocessor.joblib')
        print("Updated preprocessor.joblib with movie_averages.")

    return preprocessor, model

def predict_rating(user_id, movie_id, preprocessor, model):
    """Predicts the rating for a given user and movie."""
    user_map = preprocessor['user_map']
    movie_map = preprocessor['movie_map']
    global_mean = preprocessor['global_mean']
    movie_averages = preprocessor['movie_averages']
    
    train_matrix = model['train_matrix']
    item_sim = model['item_sim']
    
    # Cold-start baseline: Movie Average or Global Average
    baseline = movie_averages.get(movie_id, global_mean)
    
    if user_id not in user_map or movie_id not in movie_map:
        return baseline
    
    u_idx = user_map[user_id]
    m_idx = movie_map[movie_id]
    
    # Prediction logic: Weighted average of user's ratings for similar items
    sim_scores = item_sim[m_idx]
    user_ratings = train_matrix[u_idx, :]
    
    rated_indices = np.where(user_ratings > 0)[0]
    
    if len(rated_indices) == 0:
        return baseline
    
    relevant_sims = sim_scores[rated_indices]
    relevant_ratings = user_ratings[rated_indices]
    
    if np.sum(np.abs(relevant_sims)) == 0:
        return baseline
        
    prediction = np.sum(relevant_sims * relevant_ratings) / np.sum(np.abs(relevant_sims))
    return np.clip(prediction, 0.5, 5.0)

def get_recommendations(user_id, preprocessor, model, n=5):
    """Generates Top N recommendations for a user."""
    user_map = preprocessor['user_map']
    movie_map = preprocessor['movie_map']
    inv_movie_map = {i: id for id, i in movie_map.items()}
    train_matrix = model['train_matrix']
    
    if user_id not in user_map:
        return "User not found."
    
    u_idx = user_map[user_id]
    already_rated = train_matrix[u_idx, :] > 0
    unrated_m_indices = np.where(~already_rated)[0]
    
    preds = []
    for m_idx in unrated_m_indices:
        # Use existing predict logic
        movie_id = inv_movie_map[m_idx]
        score = predict_rating(user_id, movie_id, preprocessor, model)
        preds.append((movie_id, score))
    
    preds.sort(key=lambda x: x[1], reverse=True)
    return preds[:n]

if __name__ == "__main__":
    pre, mdl = load_recommender()
    
    uid = 1
    mid = 1
    
    pred = predict_rating(uid, mid, pre, mdl)
    print(f"Predicted Rating for User {uid} on Movie {mid}: {pred:.2f}")
    
    print(f"\nTop 5 recommendations for User {uid}:")
    recs = get_recommendations(uid, pre, mdl, n=5)
    for m, s in recs:
        print(f"Movie ID: {m}, Predicted Score: {s:.2f}")
