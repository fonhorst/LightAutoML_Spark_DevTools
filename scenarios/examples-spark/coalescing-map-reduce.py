import functools
import logging.config
import logging.config
import math
import os

from pyspark.sql import functions as sf
from sparklightautoml.computations.manager import get_executors, PrefferedLocsPartitionCoalescerTransformer
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.tasks.base import SparkTask
from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT, JobGroup

from examples_utils import get_spark_session, get_dataset

logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename='/tmp/slama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def main():
    spark = get_spark_session()
    parallelism = int(os.environ.get("EXP_JOB_PARALLELISM", "2"))
    feat_pipe = "lgb_adv"
    dataset_name = os.environ.get("DATASET", "lama_test_dataset")
    dataset = get_dataset(dataset_name)
    dataset_path = f"file:///opt/spark_data/preproccessed_datasets/{dataset_name}__{feat_pipe}__features.dataset"

    # load and prepare data
    ds = SparkDataset.load(
        path=dataset_path,
        persistence_manager=PlainCachePersistenceManager()
    )

    execs = get_executors()
    # exec_cores = get_executors_cores()
    execs_per_slot = max(1, math.floor(len(execs) / parallelism))
    slots_num = int(len(execs) / execs_per_slot)

    coalesce_dfs = []
    pref_locs_for_dfs = []
    for i in range(slots_num):
        pref_locs = execs[i * execs_per_slot: (i + 1) * execs_per_slot]
        logger.info(f"Slot num #{i}, pref locs: {pref_locs}")
        coalesced_data = PrefferedLocsPartitionCoalescerTransformer(pref_locs=pref_locs) \
            .transform(ds.data).cache()
        # coalesced_data.write.format("noop").mode("overwrite").save()
        coalesce_dfs.append(coalesced_data)
        pref_locs_for_dfs.append(pref_locs)

    # a trick for execution optimization
    all_df = functools.reduce(lambda acc, df: acc.unionByName(df), coalesce_dfs)
    all_df.count()

    # working with aggregating
    for i, df in enumerate(coalesce_dfs):
        logger.info(f"Calculating for slot #{i}")
        with JobGroup(f"Agg slot #{i}", f"Should be executed on {pref_locs_for_dfs[i]}", spark):
            # df.agg(sf.count('*').alias('count')).write.format('noop').mode('overwrite').save()
            df.groupby(dataset.roles['target']).agg(sf.count('*').alias('count')).write.format('noop').mode('overwrite').save()

    # working with Tree Aggregating
    for i, df in enumerate(coalesce_dfs):
        logger.info(f"Calculating for slot #{i}")
        with JobGroup(f"TreeAgg slot #{i}", f"Should be executed on {pref_locs_for_dfs[i]}", spark):
            score = SparkTask(dataset.task_type).get_dataset_metric()
            metric_value = score(
                df.select(
                    sf.lit(0).alias(SparkDataset.ID_COLUMN),
                    sf.col(dataset.roles['target']).alias('target'),
                    sf.rand(seed=42).alias('prediction')
                )
            )

            logger.info(f"Metric value: {metric_value}")

    spark.stop()


if __name__ == "__main__":
    main()