import math

from pyspark.sql import SparkSession
from sparklightautoml.computations.manager import get_executors_cores, get_executors

import numpy as np


def main():
    spark = SparkSession.builder.getOrCreate()
    num_slots = get_executors_cores() * len(get_executors())

    df = spark.read.parquet("dataset.parquet").repartition(num_slots).cache()
    df.write.mode('overwrite').format('noop').save()

    for proportion in np.arange(0.1, 1.1, 0.1):
        new_num_slots = max(1, math.floor(num_slots * proportion))
        # TODO: measure exec time and report to mlflow
        # TODO: params to mlflow (dataset name, executors count and cores count, app_id, num_slots, new_num_slots and etc.)
        df.repartition(new_num_slots).write.mode('overwrite').format('noop').save()
    pass


if __name__ == "__main__":
    main()
