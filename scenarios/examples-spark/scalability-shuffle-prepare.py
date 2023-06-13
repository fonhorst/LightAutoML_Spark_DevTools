import logging.config
import os

import mlflow
from sparklightautoml.computations.base import ComputationSlot
from sparklightautoml.computations.utils import get_executors
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT
from examples_utils import get_spark_session, train_test_split, ReportingParallelComputionsManager, get_ml_algo, \
    check_allocated_executors

logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename='/tmp/slama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("In the very beginning of parallel-optuna")
    spark = get_spark_session()

    dataset_name = os.environ.get("DATASET", "lama_test_dataset")
    parallelism = int(os.environ.get("EXP_JOB_PARALLELISM", "4"))
    n_trials = 64
    timeout = 60000
    stabilize = False
    feat_pipe, default_params, ml_algo = get_ml_algo()
    dataset_path = f"file:///opt/spark_data/preproccessed_datasets/{dataset_name}__{feat_pipe}__features.dataset"

    with mlflow.start_run(experiment_id=os.environ["EXPERIMENT"]):
        mlflow.log_params({
            "app_id": spark.sparkContext.applicationId,
            "app_name": spark.sparkContext.appName,
            "parallelism": parallelism,
            "dataset": dataset_name,
            "dataset_path": dataset_path,
            "feat_pipe": feat_pipe
        })
        mlflow.log_dict(dict(spark.sparkContext.getConf().getAll()), "spark_conf.json")

        # load and prepare data
        ds = SparkDataset.load(
            path=dataset_path,
            persistence_manager=PlainCachePersistenceManager()
        )

        check_allocated_executors()

        train_ds, test_ds = train_test_split(ds, test_slice_or_fold_num=4)

        # create main entities
        computations_manager_optuna = \
            ReportingParallelComputionsManager(parallelism=parallelism, use_location_prefs_mode=True)

        def build_func(seq_id: int):
            def _func(slot: ComputationSlot):
                logger.info(f"Running mock task: #{seq_id}")
                slot.dataset.data.write.mode("overwrite").format("noop").save()
                logger.info(f"Finished mock task: #{seq_id}")
            return _func
        tasks = [build_func(i) for i in range(parallelism)]

        computations_manager_optuna.compute_on_dataset(train_ds, tasks)

        prep_ds_size = computations_manager_optuna.last_session.prepare_dataset_time
        mlflow.log_metric("optuna_prepare_dataset_pref_locs_time", prep_ds_size or 0.0)

    k = 0
