from pygemstones.util import log as l
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from modules import fnc_general as fgen
from modules.cl_recommendation import Recommendation


def create_tag_embeddings(products_ds):
    # combine all tags into a string for each product
    products_ds["tags_joined"] = products_ds["tags"].apply(
        lambda x: " ".join(x.split("|"))
    )

    # create TF-IDF matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(products_ds["tags_joined"])

    return tfidf_matrix, vectorizer


# recommend products based on TF-IDF similarity using kNN
def recommend_products_tf_idf(product_id, products_ds, tfidf_matrix, k=10):
    # check if the product ID is in the dataset
    if product_id not in products_ds["product_id"].values:
        l.e(f"Product with ID {product_id} not found.")
        return

    # adjust the number of neighbors based on the dataset size
    n_samples = tfidf_matrix.shape[0]
    k = min(k, n_samples - 1)

    # create and fit the kNN model
    model_knn = NearestNeighbors(
        metric="cosine", algorithm="brute", n_neighbors=k + 1, n_jobs=-1
    )
    model_knn.fit(tfidf_matrix)

    # get the index of the product in the dataset
    product_idx = products_ds.index[products_ds["product_id"] == product_id].tolist()[0]
    product_title = products_ds.loc[
        products_ds["product_id"] == product_id, "title"
    ].values[0]

    # find the nearest neighbors
    distances, indices = model_knn.kneighbors(
        tfidf_matrix[product_idx], n_neighbors=k + 1
    )

    # prepare the recommendations list
    recommendations = []
    for i, index in enumerate(indices.flatten()[1:]):
        if index < len(products_ds):
            similar_product_id = products_ds.iloc[index]["product_id"]
            similar_product_title = products_ds.iloc[index]["title"]
            distance = distances.flatten()[i + 1]
            recommendations.append(
                Recommendation(similar_product_id, similar_product_title, distance)
            )
        else:
            l.e(f"Index {index} is out of bounds for products dataset.")

    # display recommendations
    fgen.show_recommendations(
        f'Recommendations for "{product_title}" (lower is better):', recommendations
    )
