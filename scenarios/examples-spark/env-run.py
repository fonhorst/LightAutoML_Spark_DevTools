from pyspark.sql import SparkSession
import os
import pprint

spark = SparkSession.builder.getOrCreate()

pprint.pprint(os.environ)

spark.stop()
