import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

from modules.cl_recommendation import Recommendation


# function to calculate similarity matrix
def calculate_similarity_matrix(products_df):
    # combine title and category using TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(
        products_df["title"] + " " + products_df["category"]
    )

    # convert tag lists into a binary matrix
    mlb = MultiLabelBinarizer()
    tag_matrix = mlb.fit_transform(products_df["tag_list"])

    # combine tfidf matrix and tag matrix
    combined_features = np.hstack([tfidf_matrix.toarray(), tag_matrix])

    # calculate cosine similarity
    similarity_matrix = cosine_similarity(combined_features)

    return similarity_matrix


# recommendation functions
def recommend_similar_products(
    products_df, product_id, product_to_index, similarity_matrix, top_k=5
):
    product_idx = product_to_index[product_id]
    similarities = similarity_matrix[product_idx]
    similar_indices = np.argsort(similarities)[::-1][1 : top_k + 1]
    recommendations = []

    for idx in similar_indices:
        similar_id = list(product_to_index.keys())[
            list(product_to_index.values()).index(idx)
        ]

        product_info = products_df[products_df["product_id"] == similar_id].iloc[0]
        title = product_info["title"]
        category = product_info["category"]
        distance = similarities[idx] if idx < len(similarities) else None
        recommendations.append(Recommendation(similar_id, title, distance))

    return recommendations


def recommend_products_for_user(
    products_df,
    user_id,
    ratings_df,
    similarity_matrix,
    product_to_index,
    show_consumed,
    top_k=5,
):
    # get ratings for the specified user
    user_ratings = ratings_df[ratings_df["user_id"] == user_id]

    # determine products to consider based on show_consumed flag
    if show_consumed:
        considered_products = set(
            user_ratings[user_ratings["consumed"] == 1]["product_id"]
        )
    else:
        all_products = set(product_to_index.keys())
        consumed_products = set(
            user_ratings[user_ratings["consumed"] == 1]["product_id"]
        )
        considered_products = all_products - consumed_products

    # filter indices of considered products
    considered_indices = [
        product_to_index[pid] for pid in considered_products if pid in product_to_index
    ]

    # calculate average similarity for considered products
    user_rated_indices = [
        product_to_index[pid]
        for pid in user_ratings["product_id"]
        if pid in product_to_index
    ]
    user_similarity_matrix = similarity_matrix[user_rated_indices][
        :, considered_indices
    ]
    avg_similarities = np.mean(user_similarity_matrix, axis=0)

    # sort and select top_k products based on average similarity
    top_indices = np.argsort(avg_similarities)[::-1][:top_k]
    top_product_ids = [
        list(product_to_index.keys())[
            list(product_to_index.values()).index(considered_indices[idx])
        ]
        for idx in top_indices
    ]

    # create recommendations
    recommendations = [
        Recommendation(
            pid,
            products_df.loc[products_df["product_id"] == pid, "title"].iloc[0],
            avg_similarities[considered_indices.index(product_to_index[pid])]
            if considered_indices.index(product_to_index[pid]) < len(avg_similarities)
            else None,
        )
        for pid in top_product_ids
    ]

    return recommendations


def recommend_similar_for_user(
    products_df,
    product_ids,
    user_id,
    product_id,
    product_to_index,
    similarity_matrix,
    ratings_df,
    show_consumed,
    top_k=5,
):
    similar_products = recommend_similar_products(
        products_df, product_id, product_to_index, similarity_matrix, top_k
    )

    user_rated_products = recommend_products_for_user(
        products_df,
        user_id,
        ratings_df,
        similarity_matrix,
        product_to_index,
        show_consumed,
        len(product_ids),
    )

    # filter similar products to those rated by the user
    filtered_similar_products = [
        prod
        for prod in similar_products
        if prod.id in [p.id for p in user_rated_products]
    ]

    return filtered_similar_products[:top_k]
