#!/usr/bin/python3

import time
import os
import mlflow
from sparklightautoml.utils import SparkDataFrame

from pyspark.sql import SparkSession

if os.environ.get("SCRIPT_ENV", None) == "cluster":
    spark = SparkSession.builder.getOrCreate()
# spark = SparkSession.builder.getOrCreate()

pythonpath = os.environ.get("PYTHONPATH", None)
print(f"PYTHONPATH: {pythonpath}")

result = spark.sparkContext.parallelize([i for i in range(10)]).sum()
print(f"Test result: {result}")

spark.stop()
