import pandas as pd
from pygemstones.util import log as l

from modules import fnc_hybrid as fnc

# general
is_debug = True

# load dataset
ratings_ds = pd.read_csv("extras/data/ratings.csv")
products_ds = pd.read_csv("extras/data/products.csv")

if is_debug:
    l.m("Ratings Head:")
    l.m(ratings_ds.head(10))

    l.m("Products Head:")
    l.m(products_ds.head(10))

# stats
n_ratings = len(ratings_ds)
n_products = len(ratings_ds["product_id"].unique())
n_users = len(ratings_ds["user_id"].unique())

if is_debug:
    l.d(f"Number of ratings: {n_ratings}")
    l.d(f"Number of unique products: {n_products}")
    l.d(f"Number of unique users: {n_users}")
    l.d(f"Average ratings per user: {round(n_ratings/n_users, 2)}")
    l.d(f"Average ratings per product: {round(n_ratings/n_products, 2)}")

# recommended products by user
user_id = 1
fnc.hybrid_recommendations(
    user_id,
    10,
    ratings_ds,
    products_ds,
)
