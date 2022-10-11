import logging
import time
import uuid
from contextlib import contextmanager
from random import random
from threading import Thread
from typing import List

from pandas import Series
from pyspark import SparkContext, StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import DataFrame as SparkDataFrame, Column
from pyspark.sql.column import _to_java_column
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.util import InheritableThread, inheritable_thread_target

from pyspark.sql import functions as F

BUCKET_NUMS = 2


logger = logging.getLogger()

spark = (
    SparkSession
    .builder
    .master('local-cluster[4,1,1024]')
    .config("spark.jars", "spark-lightautoml_2.12-0.1.jar")
    .config('spark.sql.autoBroadcastJoinThreshold', '-1')
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .config("spark.sql.shuffle.partitions", f"{BUCKET_NUMS}")
    .config("spark.locality.wait", "0s")
    .config("spark.scheduler.mode", "FAIR")
    .getOrCreate()
)
#
# spark = (
#     SparkSession
#     .builder
#     .getOrCreate()
# )

def delay_scala_udf(col):
    sc = SparkContext._active_spark_context
    return Column(
        sc._jvm.org.apache.spark.sql.lightautoml.functions.delay_scala_udf(_to_java_column(col)))


@contextmanager
def JobGroup(group_id: str, description: str):
    sc = SparkSession.getActiveSession().sparkContext
    sc.setJobGroup(group_id, description)
    yield
    sc._jsc.clearJobGroup()


def make_initial_dataset(col_count: int):
    with JobGroup("initial dataset", "Creating the initial dataset"):
        data = [{"id": str(uuid.uuid4()), **{f"col_{i}": random() for i in range(col_count)}} for i in range(100)]
        df = spark.createDataFrame(data)
    return df


def cache_and_materialize(df: SparkDataFrame, type: str = "cache") -> SparkDataFrame:
    if type == 'cache':
        df = df.cache()
    elif type == 'locchkp':
        df = df.localCheckpoint(eager=True)
    else:
        raise Exception(f"Unknown type: {type}")

    df.write.mode('overwrite').format('noop').save()
    return df


@pandas_udf("float")
def func_rand(s: Series) -> Series:
    time.sleep(60)
    return s.map(lambda x: random())


@inheritable_thread_target
def target_func(df: SparkDataFrame, col_name: str, i: int, sdfs: List) -> SparkDataFrame:
    # df = df.localCheckpoint(eager=True)
    # df = spark.createDataFrame(df.rdd, schema=df.schema)

    # sdf = df.select('*', func_rand(col_name).alias('col_0_rand')).cache()
    # sdf = df.select('*', F.rand(42).alias('col_0_rand')).cache()
    sdf = df.select('*', delay_scala_udf(col_name).alias(f'col_{i}_rand')).cache()
    sdf.write.mode('overwrite').format('noop').save()

    sdfs[i] = sdf
    return sdf


def bucketize_table(name: str, df: SparkDataFrame) -> SparkDataFrame:
    with JobGroup("bucketize table", f"Bucketizing table {name}"):
        (
            df.write.mode('overwrite')
            .bucketBy(BUCKET_NUMS, 'id').sortBy('id')
            .saveAsTable(name, format='parquet', path=f"/tmp/{name}")
        )
        df_bucketed = spark.table(name)
    return df_bucketed


def main():
    df = make_initial_dataset(10)
    # df = df.repartition(BUCKET_NUMS)
    df = bucketize_table("base", df)

    df = cache_and_materialize(df)

    sdfs = [None, None]

    sdf_1_thread = Thread(target=target_func, args=(df, 'col_0', 0, sdfs))
    sdf_2_thread = Thread(target=target_func, args=(df, 'col_1', 1, sdfs))

    sdf_1_thread.start()
    sdf_2_thread.start()

    sdf_1_thread.join()
    sdf_2_thread.join()

    sdf_1, sdf_2 = sdfs[0], sdfs[1]

    final_sdf = sdf_1.join(sdf_2, on='id', how='inner')
    cache_and_materialize(final_sdf)

    print("Everything has been calculated")


if __name__ == '__main__':
    main()

    time.sleep(600)

    # to measure number of allocated executors
    sc = spark._jsc.sc()
    n_workers = len([executor.host() for executor in sc.statusTracker().getExecutorInfos()]) - 1

    n_desired_workers = spark.conf.get("spark.executor.instances")

    if n_workers != n_desired_workers:
        raise NotEnoughExecutorsException(...)
