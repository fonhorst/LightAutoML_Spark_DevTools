import logging.config
import os

import mlflow
from sparklightautoml.computations.utils import get_executors
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT
from examples_utils import get_spark_session, train_test_split, ReportingParallelComputionsManager, get_ml_algo

logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename='/tmp/slama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("In the very beginning of parallel-optuna")
    spark = get_spark_session()

    dataset_name = os.environ.get("DATASET", "lama_test_dataset")
    parallelism = int(os.environ.get("EXP_JOB_PARALLELISM", "1"))
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

        allocated_executos_num = len(get_executors())
        assert allocated_executos_num == int(spark.conf.get("spark.executor.instances")), \
            f"Not all desired executors have been allocated: " \
            f"{allocated_executos_num} != {spark.conf.get('spark.executor.instances')}"

        train_ds, test_ds = train_test_split(ds, test_slice_or_fold_num=4)

        # create main entities
        computations_manager_optuna = \
            ReportingParallelComputionsManager(parallelism=parallelism, use_location_prefs_mode=True)

        tasks = [lambda: logger.info(f"Running mock task: #{i}") for i in range(parallelism)]

        computations_manager_optuna.compute(tasks)

        mlflow.log_metric(
            "optuna_prepare_dataset_pref_locs_time",
            computations_manager_optuna.last_session.prepare_dataset_time
        )
