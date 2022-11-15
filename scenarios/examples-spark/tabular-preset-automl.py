import logging.config
import os
import uuid

import mlflow
import pyspark.sql.functions as sf
from pyspark.ml import PipelineModel
from sparklightautoml.automl.presets.tabular_presets import SparkTabularAutoML
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.tasks.base import SparkTask
from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT, log_exec_timer as regular_log_exec_timer

from examples_utils import get_dataset_attrs, prepare_test_and_train, get_spark_session, \
    mlflow_log_exec_timer as log_exec_timer, mlflow_deco, log_session_params_to_mlflow, check_executors_count
from examples_utils import get_persistence_manager

uid = uuid.uuid4()
logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename=f'/tmp/slama-{uid}.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)

# NOTE! This demo requires datasets to be downloaded into a local folder.
# Run ./bin/download-datasets.sh to get required datasets into the folder.


@mlflow_deco
def main(cv: int, seed: int, dataset_name: str):
    spark = get_spark_session()

    assert spark is not None

    log_session_params_to_mlflow()
    check_executors_count()

    # Algos and layers to be used during automl:
    # For example:
    # 1. use_algos = [["lgb"]]
    # 2. use_algos = [["lgb_tuned"]]
    # 3. use_algos = [["linear_l2"]]
    # 4. use_algos = [["lgb", "linear_l2"], ["lgb"]]
    use_algos = [["lgb", "linear_l2"], ["lgb"]]
    path, task_type, roles, dtype = get_dataset_attrs(dataset_name)

    persistence_manager = get_persistence_manager()
    # Alternative ways to define persistence_manager
    # persistence_manager = get_persistence_manager("CompositePlainCachePersistenceManager")
    # persistence_manager = CompositePlainCachePersistenceManager(bucket_nums=BUCKET_NUMS)

    with log_exec_timer("full"):
        with log_exec_timer("fit") as train_timer:
            task = SparkTask(task_type)
            train_data, test_data = prepare_test_and_train(spark, path, seed)

            test_data_dropped = test_data

            # optionally: set 'convert_to_onnx': True to use onnx-based version of lgb's model transformer
            automl = SparkTabularAutoML(
                spark=spark,
                task=task,
                timeout=10000,
                general_params={"use_algos": use_algos},
                lgb_params={
                    'default_params': {'numIterations': 500},
                    'freeze_defaults': True,
                    'use_single_dataset_mode': True,
                    'convert_to_onnx': False,
                    'mini_batch_size': 1000
                },
                linear_l2_params={'default_params': {'regParam': [1e-5]}},
                reader_params={"cv": cv, "advanced_roles": False},
                config_path="tabular_config.yml"
            )

            oof_predictions = automl.fit_predict(
                train_data,
                roles=roles,
                persistence_manager=persistence_manager
            )

        logger.info("Predicting on out of fold")

        score = task.get_dataset_metric()
        metric_value = score(oof_predictions)

        logger.info(f"score for out-of-fold predictions: {metric_value}")

        mlflow.log_metric("oof_score", metric_value)

        transformer = automl.transformer()

        oof_predictions.unpersist()
        # this is necessary if persistence_manager is of CompositeManager type
        # it may not be possible to obtain oof_predictions (predictions from fit_predict) after calling unpersist_all
        automl.persistence_manager.unpersist_all()

        with log_exec_timer("predict") as predict_timer:
            te_pred = automl.predict(test_data_dropped, add_reader_attrs=True)

            score = task.get_dataset_metric()
            test_metric_value = score(te_pred)

            logger.info(f"score for test predictions: {test_metric_value}")
            mlflow.log_metric("test_score", test_metric_value)

        with regular_log_exec_timer("spark-lama predicting on test (#2 way)"):
            te_pred = automl.transformer().transform(test_data_dropped)

            pred_column = next(c for c in te_pred.columns if c.startswith('prediction'))
            score = task.get_dataset_metric()
            test_metric_value = score(te_pred.select(
                SparkDataset.ID_COLUMN,
                sf.col(roles['target']).alias('target'),
                sf.col(pred_column).alias('prediction')
            ))

            logger.info(f"score for test predictions: {test_metric_value}")

        base_path = f"/tmp/models/tabular-preset-automl-{uid}"
        automl_model_path = os.path.join(base_path, "automl_pipeline")
        os.makedirs(base_path, exist_ok=True)

        with log_exec_timer("model_saving") as saving_timer:
            transformer.write().overwrite().save(automl_model_path)

        with log_exec_timer("model_loading") as loading_timer:
            pipeline_model = PipelineModel.load(automl_model_path)

        with regular_log_exec_timer("spark-lama predicting on test (#3 way)"):
            te_pred = pipeline_model.transform(test_data_dropped)

            pred_column = next(c for c in te_pred.columns if c.startswith('prediction'))
            score = task.get_dataset_metric()
            test_metric_value = score(te_pred.select(
                SparkDataset.ID_COLUMN,
                sf.col(roles['target']).alias('target'),
                sf.col(pred_column).alias('prediction')
            ))

    logger.info(f"score for test predictions via loaded pipeline: {test_metric_value}")

    logger.info("Predicting is finished")

    result = {
        "seed": seed,
        "dataset": dataset_name,
        "used_algo": str(use_algos),
        "metric_value": metric_value,
        "test_metric_value": test_metric_value,
        "train_duration_secs": train_timer.duration,
        "predict_duration_secs": predict_timer.duration,
        "saving_duration_secs": saving_timer.duration,
        "loading_duration_secs": loading_timer.duration
    }

    print(f"EXP-RESULT: {result}")

    train_data.unpersist()
    test_data.unpersist()

    check_executors_count()

    spark.stop()

    return result


if __name__ == "__main__":
    main()
