import logging.config
import uuid

import mlflow
from sparklightautoml.automl.presets.tabular_presets import SparkTabularAutoML
from sparklightautoml.tasks.base import SparkTask
from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT

from examples_utils import get_persistence_manager
from examples_utils import prepare_test_and_train, get_spark_session, mlflow_deco, \
    mlflow_log_exec_timer, log_session_params_to_mlflow

logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename='/tmp/slama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)

# NOTE! This demo requires datasets to be downloaded into a local folder.
# Run ./bin/download-datasets.sh to get required datasets into the folder.

uid = uuid.uuid4()
logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename=f'/tmp/slama-{uid}.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


@mlflow_deco
def main(cv: int, seed: int, dataset_name: str):
    spark = get_spark_session()

    assert spark is not None

    log_session_params_to_mlflow()

    # Algos and layers to be used during automl:
    # For example:
    # 1. use_algos = [["lgb"]]
    # 2. use_algos = [["lgb_tuned"]]
    # 3. use_algos = [["linear_l2"]]
    # 4. use_algos = [["lgb", "linear_l2"], ["lgb"]]
    # use_algos = [["lgb", "linear_l2"], ["lgb"]]
    use_algos = [["lgb"]]
    # path, task_type, roles, dtype = get_dataset_attrs(dataset_name)

    # path = '/opt/experiments/test_exp/full_second_level_train.parquet'
    path = 'file:///opt/spark_data/replay/experiments/ml25m_first_level_default/partial_train_replay__models__slim__SLIM_2e7686b8f7124e5d9289c83f1071549d.parquet'
    task_type = 'binary'
    roles = {"target": "target"}

    persistence_manager = get_persistence_manager(run_id=str(uid))
    # Alternative ways to define persistence_manager
    # persistence_manager = get_persistence_manager("CompositePlainCachePersistenceManager")
    # persistence_manager = CompositePlainCachePersistenceManager(bucket_nums=BUCKET_NUMS)

    with mlflow_log_exec_timer("full") as train_timer:
        task = SparkTask(task_type)
        train_data, test_data = prepare_test_and_train(spark, path, seed, is_csv=False)

        train_data = train_data.drop("user_idx", "item_idx")
        test_data_dropped = test_data

        # optionally: set 'convert_to_onnx': True to use onnx-based version of lgb's model transformer
        automl = SparkTabularAutoML(
            spark=spark,
            task=task,
            timeout=10000,
            general_params={"use_algos": use_algos},
            lgb_params={
                'use_single_dataset_mode': True,
                'convert_to_onnx': False,
                'mini_batch_size': 1000
            },
            linear_l2_params={'default_params': {'regParam': [1e-5]}},
            reader_params={"cv": cv, "advanced_roles": False},
            config_path="tabular_config.yml"
        )

        with mlflow_log_exec_timer("fit"):
            oof_predictions = automl.fit_predict(
                train_data,
                roles=roles,
                persistence_manager=persistence_manager
            )

            logger.info("Predicting on out of fold")

            score = task.get_dataset_metric()
            metric_value = score(oof_predictions)

            mlflow.log_metric()

        logger.info(f"score for out-of-fold predictions: {metric_value}")

        # transformer = automl.transformer()

        oof_predictions.unpersist()
        # this is necessary if persistence_manager is of CompositeManager type
        # it may not be possible to obtain oof_predictions (predictions from fit_predict) after calling unpersist_all
        automl.persistence_manager.unpersist_all()

        with mlflow_log_exec_timer("predict"):
            te_pred = automl.transformer().transform(test_data_dropped)

            score = task.get_dataset_metric()
            test_metric_value = score(te_pred)

            logger.info(f"score for test predictions: {test_metric_value}")

    logger.info("Predicting is finished")

    train_data.unpersist()
    test_data.unpersist()

    spark.stop()


if __name__ == "__main__":
    main()
