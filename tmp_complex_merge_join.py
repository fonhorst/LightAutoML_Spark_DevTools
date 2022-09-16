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

BUCKET_NUMS = 100
INITIAL_COL_COUNT = 10
FP_COLS_COUNT = INITIAL_COL_COUNT
CV_COLS_COUNT = 10
CV_FOLDS = 5

logger = logging.getLogger()

spark = (
    SparkSession
    .builder
    .master('local[2]')
    .config("spark.jars", "spark-lightautoml_2.12-0.1.jar")
    .config('spark.sql.autoBroadcastJoinThreshold', '-1')
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .config("spark.sql.shuffle.partitions", "100")
    .getOrCreate()
)


def test_scala_udf(col):
    sc = SparkContext._active_spark_context
    return Column(
        sc._jvm.org.apache.spark.sql.lightautoml.functions.test_scala_udf(_to_java_column(col)))


@pandas_udf("float")
def func_rand(s: Series) -> Series:
    return s.map(lambda x: random())


@pandas_udf("float")
def func_add_up(s: Series) -> Series:
    return s.map(lambda x: x + 2)


@pandas_udf("float")
def func_div(s: Series) -> Series:
    return s.map(lambda x: x / 3)


@contextmanager
def log_exec_time(name: Optional[str] = None, write_log=True):

    start = datetime.now()

    yield

    end = datetime.now()
    duration = (end - start).total_seconds()

    msg = f"Exec time of {name}: {duration}" if name else f"Exec time: {duration}"
    logger.warning(msg)


@contextmanager
def JobGroup(group_id: str, description: str):
    sc = SparkSession.getActiveSession().sparkContext
    sc.setJobGroup(group_id, description)
    yield
    sc._jsc.clearJobGroup()


def join_multiple(dfs: List[SparkDataFrame]) -> SparkDataFrame:
    return functools.reduce(lambda x_df, y_df: x_df.join(y_df, on='id', how='inner'), dfs)


def cache_and_materialize(df: SparkDataFrame, type: str = "cache") -> SparkDataFrame:
    if type == 'cache':
        df = df.cache()
    elif type == 'locchkp':
        df = df.localCheckpoint(eager=True)
    else:
        raise Exception(f"Unknown type: {type}")

    df.write.mode('overwrite').format('noop').save()
    return df


def make_initial_dataset(col_count: int):
    with JobGroup("initial dataset", "Creating the initial dataset"):
        data = [{"id": str(uuid.uuid4()), **{f"col_{i}": random() for i in range(col_count)}} for i in range(100)]
        df = spark.createDataFrame(data)
    return df


def make_new_columns(df: SparkDataFrame, new_col_prefix: str, col_count: int, use_python_udf: bool = False):
    # def make_transformer(col_name: str, i: int):
    #     return func_div(func_add_up(func_rand(col_name))).alias(f"{new_col_prefix}_{i}")

    if use_python_udf:
        new_df = df.select('id', *[func_rand('id').alias(f"{new_col_prefix}_{i}") for i in range(col_count)])
    else:
        new_df = df.select('id', *[test_scala_udf('id').alias(f"{new_col_prefix}_{i}") for i in range(col_count)])

    return cache_and_materialize(new_df)


def bucketize_table(name: str, df: SparkDataFrame) -> SparkDataFrame:
    with JobGroup("bucketize table", f"Bucketizing table {name}"):
        (
            df.write.mode('overwrite')
            .bucketBy(BUCKET_NUMS, 'id').sortBy('id')
            .saveAsTable(name, format='parquet', path=f"/tmp/{name}")
        )
        df_bucketed = spark.table(name)
    return df_bucketed


# TODO: how to know columns?
def load_bucketed_table(name: str) -> SparkDataFrame:
    with JobGroup("load bucketed table", f"Loading bucketed table {name}"):
        spark.sql(f"""
            CREATE TABLE {name}
            (id long, d long, e string)
            USING PARQUET 
            CLUSTERED BY (id) SORTED BY (id) INTO {BUCKET_NUMS} BUCKETS
            LOCATION '/tmp/{name}'
            """.strip())
        df_bucketed = spark.table(name)
    return df_bucketed


def make_fp(name: str, df: SparkDataFrame, cols_count: int) -> SparkDataFrame:
    with JobGroup("make fp", f"Processing FP {name}"):
        df_layer_0 = make_new_columns(df, f"{name}_layer0", cols_count)
        df_layer_1 = make_new_columns(df_layer_0, f"{name}_layer1", cols_count)
        df_layer_2 = make_new_columns(df_layer_1, f"{name}_layer2", cols_count)

        dfs = [df_layer_0, df_layer_1, df_layer_2]

        joined_df = join_multiple(dfs)

    return joined_df


# TODO: folds should be filled only partially
def make_crossval(name: str, df: SparkDataFrame) -> SparkDataFrame:
    with JobGroup("make crossval", f"Processing CV {name}"):
        dfs = [
            make_new_columns(df, f"{name}_fold_{i}", CV_COLS_COUNT)
            for i in range(CV_FOLDS)
        ]

    joined_df = join_multiple(dfs)

    return joined_df


def scenario_only_fp():
    # preparing the base dataset
    base_df = make_initial_dataset(col_count=INITIAL_COL_COUNT)
    base_df = bucketize_table("base", base_df)
    base_df = cache_and_materialize(base_df)

    fp_df = make_fp("fp0", base_df, FP_COLS_COUNT)

    cache_and_materialize(fp_df)


def scenario_simple():
    # preparing the base dataset
    base_df = make_initial_dataset(col_count=INITIAL_COL_COUNT)
    base_df = bucketize_table("base", base_df)
    base_df = cache_and_materialize(base_df)

    with JobGroup("make fp", f"Processing new columns"):
        df_layer_0 = make_new_columns(base_df, f"new_columns_0", FP_COLS_COUNT, use_python_udf=True)
        df_layer_1 = make_new_columns(base_df, f"new_columns_0", FP_COLS_COUNT, use_python_udf=False)
        # df_layer_2 = make_new_columns(df_layer_1, f"new_columns_0", FP_COLS_COUNT)

    # works as plan truncating and makes joins bucketed
    df_layer_0 = bucketize_table("df_layer_0", df_layer_0)

    # joined_df = join_multiple([base_df, df_layer_0, df_layer_1, df_layer_2])
    joined_df = join_multiple([base_df, df_layer_0])

    with JobGroup("first join", "First join with use_python_udf=True"):
        joined_df = cache_and_materialize(joined_df)

    joined_df_2 = join_multiple([joined_df, df_layer_1])

    with JobGroup("second join", "Second join with use_python_udf=False"):
        cache_and_materialize(joined_df_2)


def scenario_regular_mlpipe():
    # preparing the base dataset
    base_df = make_initial_dataset(col_count=INITIAL_COL_COUNT)
    base_df = bucketize_table("base", base_df)
    base_df = cache_and_materialize(base_df)

    fp_df = make_fp("fp0", base_df, FP_COLS_COUNT)
    predicts_df = make_crossval("cv0", fp_df)

    cache_and_materialize(predicts_df)


def scenario_mlpipe_with_two_models():
    # preparing the base dataset
    base_df = make_initial_dataset(col_count=INITIAL_COL_COUNT)
    base_df = bucketize_table("base", base_df)
    base_df = cache_and_materialize(base_df)

    fp_df = make_fp("fp0", base_df, FP_COLS_COUNT)
    predicts_0_df = make_crossval("cv0", fp_df)
    predicts_1_df = make_crossval("cv1", fp_df)

    predicts_df = join_multiple([predicts_0_df, predicts_1_df])

    cache_and_materialize(predicts_df)


def scenario_mlpipe_with_two_fpipes(unequal: bool = True):
    # preparing the base dataset
    base_df = make_initial_dataset(col_count=INITIAL_COL_COUNT)
    base_df = bucketize_table("base", base_df)
    base_df = cache_and_materialize(base_df)

    fp_0_df = make_fp("fp0", base_df, FP_COLS_COUNT)

    fp_1_df = make_fp("fp1", base_df, FP_COLS_COUNT // 10 if unequal else FP_COLS_COUNT)

    fp_df = join_multiple([fp_0_df, fp_1_df])

    predicts_df = make_crossval("cv0", fp_df)

    cache_and_materialize(predicts_df)


def scenario_mlpipe_with_two_fpipes_and_two_models(unequal: bool = True):
    # preparing the base dataset
    base_df = make_initial_dataset(col_count=INITIAL_COL_COUNT)
    base_df = bucketize_table("base", base_df)
    base_df = cache_and_materialize(base_df)

    fp_0_df = make_fp("fp0", base_df, FP_COLS_COUNT)

    fp_1_df = make_fp("fp1", base_df, FP_COLS_COUNT // 10 if unequal else FP_COLS_COUNT)

    fp_df = join_multiple([fp_0_df, fp_1_df])

    predicts_0_df = make_crossval("cv0", fp_df)
    predicts_1_df = make_crossval("cv1", fp_df)

    predicts_df = join_multiple([predicts_0_df, predicts_1_df])

    cache_and_materialize(predicts_df)


def main():
    # scenario_regular_mlpipe()

    # scenario_mlpipe_with_two_fpipes(unequal=False)

    # scenario_mlpipe_with_two_models()

    # scenario_mlpipe_with_two_fpipes_and_two_models()

    # scenario_only_fp()

    scenario_simple()


if __name__ == "__main__":
    with log_exec_time("main"):
        main()

    time.sleep(600)

    spark.stop()
