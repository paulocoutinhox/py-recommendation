import pandas as pd

from modules import fnc_general as fgen
from modules import fnc_sim_tc_tag as fnc

# load data
products_df = pd.read_csv("extras/data/products.csv")
ratings_df = pd.read_csv("extras/data/ratings.csv")
users_df = pd.read_csv("extras/data/users.csv")

# preprocessing tags into lists
products_df["tag_list"] = products_df["tags"].apply(lambda x: x.split("|"))

# map product and user IDs to a continuous space
product_ids = products_df["product_id"].unique().tolist()
user_ids = users_df["user_id"].unique().tolist()

product_to_index = {product_id: idx for idx, product_id in enumerate(product_ids)}
user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}

# pre-calculate similarity matrix
similarity_matrix = fnc.calculate_similarity_matrix(products_df)

# example usage
product_id = 1
user_id = 1
show_consumed = False

# similar products
similar_products = fnc.recommend_similar_products(
    products_df,
    product_id,
    product_to_index,
    similarity_matrix,
)

fgen.show_recommendations("Similar Products:", similar_products)

# similar user products
products_for_user = fnc.recommend_products_for_user(
    products_df,
    user_id,
    ratings_df,
    similarity_matrix,
    product_to_index,
    show_consumed,
)

fgen.show_recommendations("Recommended Products for User:", products_for_user)

# similar products that user rated
similar_for_user = fnc.recommend_similar_for_user(
    products_df,
    product_ids,
    user_id,
    product_id,
    product_to_index,
    similarity_matrix,
    ratings_df,
    show_consumed,
    top_k=5,
)

fgen.show_recommendations("Similar Products Recommended for User:", similar_for_user)
