from pyspark.sql import SparkSession

from modules import fnc_spark as fnc

# spark session
spark = SparkSession.builder.appName("ProductRecommendation").getOrCreate()

# load data
products_df = spark.read.csv("extras/data/products.csv", header=True, inferSchema=True)
ratings_df = spark.read.csv("extras/data/ratings.csv", header=True, inferSchema=True)

# exemple
user_id = 1
num_recommendations = 5
recommended_products = fnc.recommend_products(
    user_id, products_df, ratings_df, num_recommendations, False
)

# show recommendation
for product in recommended_products:
    print(
        f"Product ID: {product['product_id']}, Total Similarity: {product['total_similarity']}"
    )
