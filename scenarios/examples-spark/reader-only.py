import logging.config
import os
import uuid

import mlflow
from sparklightautoml.reader.base import SparkToSparkReader
from sparklightautoml.tasks.base import SparkTask as SparkTask
from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT

from examples_utils import check_executors_count, \
    log_session_params_to_mlflow, mlflow_log_exec_timer as log_exec_timer, mlflow_deco, handle_if_2stage
from examples_utils import get_spark_session, get_dataset

uid = uuid.uuid4()
log_filename = f'/tmp/slama-{uid}.log'
logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename=log_filename))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)

# NOTE! This demo requires datasets to be downloaded into a local folder.
# Run ./bin/download-datasets.sh to get required datasets into the folder.


@mlflow_deco
def main(cv: int, seed: int, dataset_name: str = "lama_test_dataset"):
    spark = get_spark_session()

    log_session_params_to_mlflow()
    check_executors_count()

    dataset = get_dataset(dataset_name)

    # persistence_manager = get_persistence_manager(run_id=str(uid))

    with log_exec_timer("full_time"):
        train_df = dataset.load()

        train_df = handle_if_2stage(dataset_name, train_df)

        sreader = SparkToSparkReader(task=SparkTask(dataset.task_type), cv=cv, samples=10_000, advanced_roles=False)
        sreader.fit_read(train_df, roles=dataset.roles)

    logger.info("Finished")

    spark.stop()

    log_files = bool(int(os.environ.get("LOG_FILES_TO_MLFLOW", "0")))
    if log_files:
        for handler in logger.handlers:
            handler.flush()
        mlflow.log_artifact(log_filename, "run.log")

    check_executors_count()


if __name__ == "__main__":
    main()
