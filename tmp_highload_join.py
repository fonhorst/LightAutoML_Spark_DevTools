import functools
import logging
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from random import random
from typing import List, Optional

from pandas import Series
from pyspark import SparkContext
from pyspark.sql import DataFrame as SparkDataFrame, Column
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.column import _to_java_column
from pyspark.sql.functions import pandas_udf


spark = (
    SparkSession
    .builder
    .getOrCreate()
)

BUCKET_NUMS = 80
ID_COL = "id"


@contextmanager
def JobGroup(group_id: str, description: str):
    sc = SparkSession.getActiveSession().sparkContext
    sc.setJobGroup(group_id, description)
    yield
    sc._jsc.clearJobGroup()


def bucketize_table(name: str, df: SparkDataFrame) -> SparkDataFrame:
    with JobGroup("bucketize table", f"Bucketizing table {name}"):
        (
            df
            .repartition(BUCKET_NUMS, ID_COL)
            .write
            .mode('overwrite')
            .bucketBy(BUCKET_NUMS, ID_COL).sortBy(ID_COL)
            .saveAsTable(name, format='parquet', path=f"hdfs:///tmp/{name}")
        )
        df_bucketed = spark.table(name)
    return df_bucketed


def cache_and_materialize(df: SparkDataFrame, type: str = "cache") -> SparkDataFrame:
    if type == 'cache':
        df = df.cache()
    elif type == 'locchkp':
        df = df.localCheckpoint(eager=True)
    else:
        raise Exception(f"Unknown type: {type}")

    df.write.mode('overwrite').format('noop').save()
    return df


def test_scala_udf(col):
    sc = SparkContext._active_spark_context
    return Column(
        sc._jvm.org.apache.spark.sql.lightautoml.functions.test_scala_udf(_to_java_column(col)))


@pandas_udf("string")
def func_rand(s: Series) -> Series:
    return s.map(lambda x: x)


def make_new_columns(df: SparkDataFrame, new_col_prefix: str, use_python_udf: bool = False):
    cols_to_transform = [col for col in df.columns if col != ID_COL]

    if use_python_udf:
        new_df = df.select(
            ID_COL,
            *[func_rand(col).alias(f"{new_col_prefix}_{i}") for i, col in enumerate(cols_to_transform)]
        )
    else:
        new_df = df.select(
            ID_COL,
            *[F.col(col).alias(f"{new_col_prefix}_{i}") for i, col in enumerate(cols_to_transform)]
        )

    return cache_and_materialize(new_df)


def join_multiple(dfs: List[SparkDataFrame]) -> SparkDataFrame:
    return functools.reduce(lambda x_df, y_df: x_df.join(y_df, on=ID_COL, how='inner'), dfs)


def make_initial_dataset(col_count: int, row_count: int):
    with JobGroup("initial dataset", "Creating the initial dataset"):
        data = [{"id": str(uuid.uuid4()), **{f"col_{i}": random() for i in range(col_count)}} for i in range(row_count)]
        df = spark.createDataFrame(data)
    return df


def scenario_simple():
    # preparing the base dataset
    base_df = spark.read.parquet("file:///opt/spark_data/data_for_LE_TE_tests/1000000_rows_1000_columns_cardinality_10000_id.parquet")
    base_df = base_df.select(F.col('_id').alias('id'), *[F.col(f'_c{i}').alias(f'c{i}') for i in range(1000)])

    base_df = bucketize_table("abase", base_df)
    base_df = cache_and_materialize(base_df)

    with JobGroup("writing to hdfs abase", "Writing to hdfs abase"):
        base_df.write.parquet("hdfs:///tmp/abase_df.parquet", mode="overwrite")

    base_df_1 = spark.read.parquet("file:///opt/spark_data/data_for_LE_TE_tests/1000000_rows_1000_columns_cardinality_10000_id.parquet")
    base_df_1 = base_df_1.select(F.col('_id').alias('id'), *[F.col(f'_c{i}').alias(f'c{i}') for i in range(1000)])
    base_df_1 = cache_and_materialize(base_df_1)

    with JobGroup("make fp", f"Processing new columns"):
        df_layer_0 = make_new_columns(base_df_1, f"new_columns_true", use_python_udf=False)
        df_layer_1 = make_new_columns(base_df, f"new_columns_false", use_python_udf=True)
        df_layer_2 = make_new_columns(base_df, f"new_columns_python", use_python_udf=False)

    joined_df = join_multiple([base_df, df_layer_0])

    with JobGroup("first join", "First join with use_python_udf=False and no bucketing"):
        joined_df = cache_and_materialize(joined_df)

    joined_df_2 = join_multiple([base_df, df_layer_1])
    with JobGroup("second join", "Second join with use_python_udf=True"):
        joined_df_2 = cache_and_materialize(joined_df_2)

    joined_df_3 = join_multiple([base_df, df_layer_2])
    with JobGroup("second join", "Second join with use_python_udf=False"):
        joined_df_3 = cache_and_materialize(joined_df_3)

    with JobGroup("writing to hdfs first join", "Writing to hdfs first join"):
        joined_df.write.parquet("hdfs:///tmp/first_joined_df.parquet", mode="overwrite")

    with JobGroup("writing to hdfs second join", "Writing to hdfs second join"):
        joined_df_2.write.parquet("hdfs:///tmp/second_joined_df.parquet", mode="overwrite")

    with JobGroup("writing to hdfs second join", "Writing to hdfs third join"):
        joined_df_3.write.parquet("hdfs:///tmp/third_joined_df.parquet", mode="overwrite")


if __name__ == "__main__":
    scenario_simple()
