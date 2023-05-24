import logging.config
import math
import os
import time

from sparklightautoml.computations.manager import get_executors, PrefferedLocsPartitionCoalescerTransformer
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT

from examples_utils import get_spark_session

logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename='/tmp/slama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def main():
    spark = get_spark_session()
    parallelism = int(os.environ.get("EXP_JOB_PARALLELISM", "2"))
    feat_pipe = "lgb_adv"
    dataset_name = os.environ.get("DATASET", "lama_test_dataset")
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

    for i in range(slots_num):
        logger.info(f"Slot num #{i}")
        pref_locs = execs[i * execs_per_slot: (i + 1) * execs_per_slot]
        coalesced_data = PrefferedLocsPartitionCoalescerTransformer(pref_locs=pref_locs) \
            .transform(ds.data).cache()
        coalesced_data.write.mode('overwrite').format('noop').save()

    # sleep_time_secs = 600
    # logger.info(f"Sleeping for the timeout {sleep_time_secs} until spark stopping")
    # time.sleep(sleep_time_secs)

    spark.stop()


if __name__ == "__main__":
    main()
