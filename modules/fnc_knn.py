import numpy as np
from pygemstones.util import log as l
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from modules import fnc_general as fgen
from modules.cl_recommendation import Recommendation


# create a user-item matrix using scipy's csr_matrix
def create_matrix(df):
    n = len(df["user_id"].unique())  # number of unique users
    m = len(df["product_id"].unique())  # number of unique products

    # get unique data from dataframe
    unique_users = np.unique(df["user_id"])
    unique_products = np.unique(df["product_id"])

    # map user and product ids
    user_mapper = dict(zip(unique_users, range(n)))
    product_mapper = dict(zip(unique_products, range(m)))

    # map indices back to user and product ids
    user_inv_mapper = dict(zip(range(n), unique_users))
    product_inv_mapper = dict(zip(range(m), unique_products))

    # create a sparse user-item matrix (sparse matrix is optimized to don't store null values)
    user_index = [user_mapper[i] for i in df["user_id"]]
    product_index = [product_mapper[i] for i in df["product_id"]]

    X = csr_matrix((df["rating"], (product_index, user_index)), shape=(m, n))

    return X, user_mapper, product_mapper, user_inv_mapper, product_inv_mapper


# find similar products based on the input product
def find_similar_products(
    product_id,
    product_titles,
    X,
    k,
    product_mapper,
    product_inv_mapper,
    metric="cosine",
):
    neighbours = []

    product_ind = product_mapper[product_id]
    product_vec = X[product_ind]

    n_samples = X.shape[0]

    if k > n_samples:
        k = n_samples - 1

    # fit a k-nearest neighbors model
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)

    product_vec = product_vec.reshape(1, -1)
    neighbour = kNN.kneighbors(product_vec, return_distance=True)

    # extract neighbor information
    for i in range(1, k):
        # neighbour
        n = neighbour[1][0][i]
        distance = neighbour[0][0][i]
        product_id = product_inv_mapper[n]

        # product title
        product_title = product_titles.get(product_id, "product-not-found")

        if product_title == "product-not-found":
            l.e(f"Product with ID {product_id} not found.")

        # append neighbour
        neighbours.append(
            Recommendation(
                product_id,
                product_title,
                distance,
            )
        )

    return neighbours


# recommend products for a given user
def recommend_products_for_user(
    user_id,
    ratings_ds,
    products_ds,
    X,
    user_mapper,
    product_mapper,
    product_inv_mapper,
    k=10,
):
    df1 = ratings_ds[ratings_ds["user_id"] == user_id]

    if df1.empty:
        l.e(f"User with ID {user_id} does not exist.")

    # find the product with the highest rating for this user
    product_id = df1[df1["rating"] == max(df1["rating"])]["product_id"].iloc[0]

    # create a dictionary mapping product ids to their titles
    product_titles = dict(zip(products_ds["product_id"], products_ds["title"]))

    # find similar products
    recommendations = find_similar_products(
        product_id, product_titles, X, k, product_mapper, product_inv_mapper
    )

    product_title = product_titles.get(product_id, "product-not-found")

    if product_title == "product-not-found":
        l.e(f"Product with ID {product_id} not found.")

    # display recommendations based on similarity
    fgen.show_recommendations(
        f'Since you consume "{product_title}", you might also like (lower is better):',
        recommendations,
    )
