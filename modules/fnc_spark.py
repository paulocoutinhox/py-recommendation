import pyspark.sql.functions as F
from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, FloatType, StringType


# preprocess tags by splitting into a list and removing duplicates.
def preprocess_tags(df):
    split_udf = udf(lambda x: list(set(x.split("|"))), ArrayType(StringType()))
    return df.withColumn("tag_list", split_udf("tags"))


# calculate Jaccard Similarity between two sets of tags.
def jaccard_similarity(tags1, tags2):
    intersection = len(set(tags1).intersection(set(tags2)))
    union = len(set(tags1).union(set(tags2)))
    return intersection / union if union else 0


# recommend products based on Jaccard similarity of tags.
def recommend_products(
    user_id, products_df, ratings_df, num_recommendations, consider_consumed=False
):
    # preprocess tags in products dataframe
    products_df = preprocess_tags(products_df)

    # get products rated by user and preprocess tags
    user_rated_products = ratings_df.filter(ratings_df.user_id == user_id)
    user_rated_products = user_rated_products.join(products_df, "product_id")
    user_rated_products = preprocess_tags(user_rated_products)

    # exclude consumed products if required
    if not consider_consumed:
        consumed_product_ids = (
            user_rated_products.filter(user_rated_products.consumed == 1)
            .select("product_id")
            .rdd.flatMap(lambda x: x)
            .collect()
        )
        products_df = products_df.filter(~col("product_id").isin(consumed_product_ids))

    # calculate Jaccard similarity for each product with user rated products
    similarity_udf = udf(jaccard_similarity, FloatType())
    products_similarity = products_df.alias("p1").crossJoin(
        user_rated_products.alias("p2")
    )
    products_similarity = products_similarity.withColumn(
        "similarity", similarity_udf("p1.tag_list", "p2.tag_list")
    )

    # aggregate similarity scores and recommend top products
    total_similarity = products_similarity.groupBy("p1.product_id").agg(
        F.sum("similarity").alias("total_similarity")
    )
    recommendations = total_similarity.orderBy(
        "total_similarity", descending=True
    ).limit(num_recommendations)

    return recommendations.collect()
