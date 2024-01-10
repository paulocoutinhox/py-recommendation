import numpy as np
import pandas as pd
import sklearn as sk
from pygemstones.util import log as l

from modules import functions as fnc

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

# frequency - create a new dataset
user_freq = (
    ratings_ds[["user_id", "product_id"]].groupby("user_id").count().reset_index()
)
user_freq.columns = ["user_id", "n_ratings"]

if is_debug:
    l.m(user_freq.head(10))

# mean rating analysis
mean_rating = ratings_ds.groupby("product_id")[["rating"]].mean()

# lowest rating analysis
lowest_rated = mean_rating["rating"].idxmin()
products_ds.loc[products_ds["product_id"] == lowest_rated]

# highest rating analysis
highest_rated = mean_rating["rating"].idxmax()
products_ds.loc[products_ds["product_id"] == highest_rated]

# number of people who rated products rated product highest
products_ds[products_ds["product_id"] == highest_rated]

# number of people who rated products rated product lowest
products_ds[products_ds["product_id"] == lowest_rated]

# products stats dataset - use bayesian average
products_stats = ratings_ds.groupby("product_id")[["rating"]].agg(["count", "mean"])
products_stats.columns = products_stats.columns.droplevel()

if is_debug:
    l.m("Products Stats Head:")
    l.m(products_stats.head(10))

# train - create user-item matrix
X, user_mapper, product_mapper, user_inv_mapper, product_inv_mapper = fnc.create_matrix(
    ratings_ds
)

# recommended products
user_id = 1
fnc.recommend_products_for_user(
    user_id,
    ratings_ds,
    products_ds,
    X,
    user_mapper,
    product_mapper,
    product_inv_mapper,
    k=2,
)
