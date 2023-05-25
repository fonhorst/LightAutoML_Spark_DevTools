import os
import time

from pyspark.resource import ResourceProfileBuilder, ExecutorResourceRequests, TaskResourceRequests
from pyspark.sql import SparkSession
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.utils import JobGroup


def map_partitions(x):
    time.sleep(60)
    return [sum(1 for _ in x)]


def main():
    spark = (
        SparkSession
        .builder
        .config("spark.shuffle.service.enabled", "true")
        .config("spark.dynamicAllocation.enabled", "true")
        .config("spark.dynamicAllocation.executorIdleTimeout", "30s")
        .config("spark.dynamicAllocation.minExecutors", "1")
        .config("spark.dynamicAllocation.maxExecutors", "5")
        .getOrCreate()
    )

    rpb = ResourceProfileBuilder()
    ereq = ExecutorResourceRequests().cores(4).memory("6g").memoryOverhead("2g")
    treq = TaskResourceRequests().cpus(4)
    rpb.require(ereq)
    rpb.require(treq)
    rp = rpb.build

    feat_pipe, ml_algo_name = "lgb_adv", "lgb"
    job_parallelism = int(os.environ.get("EXP_JOB_PARALLELISM", "1"))
    dataset_name = os.environ.get("DATASET", "lama_test_dataset")
    dataset_path = f"file:///opt/spark_data/preproccessed_datasets/{dataset_name}__{feat_pipe}__features.dataset"
    # load and prepare data
    ds = SparkDataset.load(
        path=dataset_path,
        persistence_manager=PlainCachePersistenceManager()
    )

    for i in range(2):
        # Alternative 1
        with JobGroup(f"Run #{i}", f"Calculating with Resource Profile #{rp.id}", spark):
            # barrier mode is not supported with dynamic allocation
            mapped_data_rdd = ds.data.rdd.withResources(profile=rp).mapPartitions(map_partitions)
            result = mapped_data_rdd.collect()
            print(f"Result: {result}")

    # Alternative 2
    # result = spark.createDataFrame(mapped_data_rdd.map(lambda x: (x,)))
    # result.write.parquet("output.parquet")


if __name__ == "__main__":
    main()
