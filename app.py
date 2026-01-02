import streamlit as st
import pandas as pd
import numpy as np

# If your file is named "interference.py", change this import:
# from interference import load_recommender, get_recommendations, predict_rating
from inference import load_recommender, get_recommendations, predict_rating


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
)


# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_resource
def load_model_components():
    preprocessor, model = load_recommender()
    return preprocessor, model


@st.cache_data
def load_data():
    ratings = pd.read_csv("ratings.csv")   # userId, movieId, rating, timestamp
    movies = pd.read_csv("movies.csv")     # movieId, title, genres
    return ratings, movies


@st.cache_data
def compute_basic_stats(ratings: pd.DataFrame, movies: pd.DataFrame):
    n_users = ratings["userId"].nunique()
    n_movies = ratings["movieId"].nunique()
    n_ratings = len(ratings)
    sparsity = 1 - n_ratings / (n_users * n_movies)
    return n_users, n_movies, n_ratings, sparsity


# -----------------------------
# Helper: similar movies (item-based CF)
# -----------------------------
def get_similar_movies(movie_id: int, preprocessor, model, movies_df, top_k: int = 10):
    movie_map = preprocessor["movie_map"]
    item_sim = model["item_sim"]

    if movie_id not in movie_map:
        return pd.DataFrame(columns=["movieId", "Similarity", "title", "genres"])

    m_idx = movie_map[movie_id]
    sims = item_sim[m_idx].copy()

    # Exclude the movie itself
    sims[m_idx] = -np.inf

    # Top-k similar indices
    top_idx = np.argsort(sims)[-top_k:][::-1]
    sim_scores = sims[top_idx]

    inv_movie_map = {idx: mid for mid, idx in movie_map.items()}
    similar_movie_ids = [inv_movie_map[i] for i in top_idx]

    df = pd.DataFrame({"movieId": similar_movie_ids, "Similarity": sim_scores})
    df = df.merge(movies_df, on="movieId", how="left")
    return df


# -----------------------------
# Main App
# -----------------------------
def main():
    preprocessor, model = load_model_components()
    ratings, movies = load_data()

    # ====== Header ======
    st.markdown(
        """
        <h1 style="margin-bottom:0.2rem;">🎬 Movie Recommendation System</h1>
        <p style="color: #555; margin-bottom: 1rem;">
        Interactive dashboard for item-based collaborative filtering on the MovieLens dataset.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ====== Top-level stats ======
    n_users, n_movies, n_ratings, sparsity = compute_basic_stats(ratings, movies)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users", f"{n_users:,}")
    c2.metric("Movies", f"{n_movies:,}")
    c3.metric("Ratings", f"{n_ratings:,}")
    c4.metric("Matrix Sparsity", f"{sparsity * 100:.1f}%")

    st.markdown("---")

    # ====== Sidebar ======
    st.sidebar.header("Controls")

    user_ids = sorted(ratings["userId"].unique())
    selected_user = st.sidebar.selectbox("Select User ID", user_ids)

    top_n = st.sidebar.slider("Top‑N Recommendations", min_value=1, max_value=20, value=5)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model")
    st.sidebar.write(
        "- Type: Item‑Based Collaborative Filtering\n"
        "- Similarity: Cosine\n"
        "- Baseline: Movie/Global averages"
    )

    # ====== Tabs ======
    tab1, tab2, tab3 = st.tabs(
        ["📌 Recommendations", "⭐ Predict Rating", "🎞 Similar Movies"]
    )

    # -----------------------------
    # Tab 1: Recommendations
    # -----------------------------
    with tab1:
        st.subheader(f"Top‑{top_n} Recommendations for User {selected_user}")

        if st.button("Get Recommendations", key="recs_btn"):
            recs = get_recommendations(
                user_id=selected_user,
                preprocessor=preprocessor,
                model=model,
                n=top_n,
            )

            if isinstance(recs, str):
                st.error(recs)
            else:
                recs_df = pd.DataFrame(recs, columns=["movieId", "Predicted Rating"])
                recs_df = recs_df.merge(movies, on="movieId", how="left")
                recs_df = recs_df[["movieId", "title", "genres", "Predicted Rating"]]
                recs_df = recs_df.sort_values("Predicted Rating", ascending=False)

                st.dataframe(recs_df, use_container_width=True)

        with st.expander(f"Rating history for User {selected_user}", expanded=False):
            user_hist = ratings[ratings["userId"] == selected_user].merge(
                movies, on="movieId", how="left"
            )
            user_hist = user_hist[["movieId", "title", "rating", "genres"]]
            st.write(f"Total movies rated: {len(user_hist)}")
            st.dataframe(user_hist, use_container_width=True)

    # -----------------------------
    # Tab 2: Predict Single Rating
    # -----------------------------
    with tab2:
        st.subheader("Predict Rating for a Movie")

        col_a, col_b = st.columns([2, 1])

        with col_a:
            movie_titles = movies.sort_values("title")["title"].tolist()
            selected_title = st.selectbox("Select Movie", movie_titles)
            movie_id = movies.loc[movies["title"] == selected_title, "movieId"].iloc[0]

        with col_b:
            st.write("Selected Movie")
            st.write(f"**Title:** {selected_title}")
            st.write(f"**Movie ID:** {int(movie_id)}")
            genres = movies.loc[movies["movieId"] == movie_id, "genres"].iloc[0]
            st.write(f"**Genres:** {genres}")

        if st.button("Predict Rating", key="single_pred_btn"):
            pred = predict_rating(
                user_id=selected_user,
                movie_id=int(movie_id),
                preprocessor=preprocessor,
                model=model,
            )
            st.success(
                f"Predicted rating for User {selected_user} on "
                f"**'{selected_title}'**: {pred:.2f} / 5.0"
            )

    # -----------------------------
    # Tab 3: Explore Similar Movies
    # -----------------------------
    with tab3:
        st.subheader("Similar Movies (Item‑Based)")

        movie_titles = movies.sort_values("title")["title"].tolist()
        ref_title = st.selectbox("Reference Movie", movie_titles, key="sim_movie_title")
        ref_movie_id = movies.loc[movies["title"] == ref_title, "movieId"].iloc[0]

        top_k = st.slider("Number of similar movies", 5, 30, 10)

        if st.button("Show Similar Movies", key="similar_btn"):
            sim_df = get_similar_movies(
                movie_id=int(ref_movie_id),
                preprocessor=preprocessor,
                model=model,
                movies_df=movies,
                top_k=top_k,
            )

            if sim_df.empty:
                st.warning("No similarity information available for this movie.")
            else:
                sim_df = sim_df[["movieId", "title", "genres", "Similarity"]]
                st.dataframe(sim_df, use_container_width=True)


if __name__ == "__main__":
    main()