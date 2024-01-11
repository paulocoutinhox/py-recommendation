from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader

from modules import fnc_general as fgen
from modules.cl_recommendation import Recommendation


# function to get top n collaborative filtering recommendations for a user
def get_cf_recommendations(user_id, n, ratings_ds, products_ds):
    # get the products rated by the user
    user_ratings = ratings_ds[ratings_ds["user_id"] == user_id]

    # load the user ratings into a surprise dataset object
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(
        user_ratings[["user_id", "product_id", "rating"]], reader
    )

    # train the collaborative filtering model
    algo = SVD()
    trainset = data.build_full_trainset()
    algo.fit(trainset)

    # get the top n recommended products for the user
    product_ids = set(ratings_ds["product_id"].unique()) - set(
        user_ratings["product_id"].values
    )
    predictions = [
        (product_id, algo.predict(user_id, product_id).est)
        for product_id in product_ids
    ]
    top_n_predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    top_n_product_ids = [x[0] for x in top_n_predictions]
    top_n_products = products_ds[products_ds["product_id"].isin(top_n_product_ids)]

    # create recommendations
    recommendations = [
        Recommendation(row["product_id"], row["title"], None)
        for _, row in top_n_products.iterrows()
    ]
    return recommendations


# function to get top n content-based filtering recommendations for a user
def get_cb_recommendations(user_id, n, ratings_ds, products_ds):
    # build the tfidf matrix from product tags
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(products_ds["tags"])

    # get the products rated by the user
    user_ratings = ratings_ds[ratings_ds["user_id"] == user_id]

    # get similarity scores for all products
    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # get the top n recommended products for the user
    recommended_products = []
    for product_id in user_ratings["product_id"]:
        product_index = products_ds[products_ds["product_id"] == product_id].index[0]
        product_scores = list(enumerate(cosine_sim_matrix[product_index]))
        sorted_products = sorted(product_scores, key=lambda x: x[1], reverse=True)[
            1 : n + 1
        ]

        recommended_products.extend(sorted_products)

    recommended_product_ids = set(
        [products_ds.iloc[i[0]]["product_id"] for i in recommended_products]
    ) - set(user_ratings["product_id"].values)
    recommended_products_data = products_ds[
        products_ds["product_id"].isin(recommended_product_ids)
    ]

    # create recommendations
    recommendations = [
        Recommendation(row["product_id"], row["title"], None)
        for _, row in recommended_products_data.iterrows()
    ]
    return recommendations


# show top n hybrid recommendations for a user
def hybrid_recommendations(user_id, n, ratings_ds, products_ds):
    cf_recommendations = get_cf_recommendations(user_id, n, ratings_ds, products_ds)
    cb_recommendations = get_cb_recommendations(user_id, n, ratings_ds, products_ds)

    # combine recommendations and remove duplicates
    combined_recommendations_dict = {
        rec.id: rec for rec in cf_recommendations + cb_recommendations
    }

    combined_recommendations = list(combined_recommendations_dict.values())
    combined_recommendations = sorted(combined_recommendations, key=lambda x: x.title)[
        :n
    ]

    # display recommendations
    fgen.show_recommendations(
        f'Recommendations to user "{user_id}":',
        combined_recommendations,
    )

    return combined_recommendations
