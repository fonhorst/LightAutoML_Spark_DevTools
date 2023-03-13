import logging.config
import os
import uuid

import mlflow
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.ml_algo.boost_lgbm import SparkBoostLGBM
from sparklightautoml.pipelines.features.lgb_pipeline import SparkLGBSimpleFeatures
from sparklightautoml.pipelines.ml.base import SparkMLPipeline
from sparklightautoml.reader.base import SparkToSparkReader
from sparklightautoml.tasks.base import SparkTask as SparkTask
from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT
from sparklightautoml.validation.iterators import SparkFoldsIterator

from examples_utils import get_persistence_manager, check_executors_count, \
    log_session_params_to_mlflow, mlflow_log_exec_timer as log_exec_timer, mlflow_deco, handle_if_2stage
from examples_utils import get_spark_session, prepare_test_and_train, get_dataset_attrs

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

    path, task_type, roles, dtype = get_dataset_attrs(dataset_name)

    persistence_manager = get_persistence_manager(run_id=str(uid))

    with log_exec_timer("full_time"):
        train_df, test_df = prepare_test_and_train(spark, path, seed)

        train_df = handle_if_2stage(dataset_name, train_df)
        test_df = handle_if_2stage(dataset_name, test_df)

        task = SparkTask(task_type)
        score = task.get_dataset_metric()

        sreader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False, samples=10_000)
        sdataset = sreader.fit_read(train_df, roles=roles, persistence_manager=persistence_manager)

        iterator = SparkFoldsIterator(sdataset).convert_to_holdout_iterator()

        spark_ml_algo = SparkBoostLGBM(freeze_defaults=False, use_single_dataset_mode=True)
        spark_features_pipeline = SparkLGBSimpleFeatures()

        ml_pipe = SparkMLPipeline(
            ml_algos=[spark_ml_algo],
            pre_selection=None,
            features_pipeline=spark_features_pipeline,
            post_selection=None
        )

        with log_exec_timer("fit_time") as fit_time:
            oof_preds_ds = ml_pipe.fit_predict(iterator)
            oof_score = score(oof_preds_ds[:, spark_ml_algo.prediction_feature])

            logger.info(f"OOF score: {oof_score}")
            mlflow.log_metric("oof_score", oof_score)

        # 1. first way (LAMA API)
        with log_exec_timer("predict") as predict_time:
            test_sds = sreader.read(test_df, add_array_attrs=True)
            test_preds_ds = ml_pipe.predict(test_sds)
            test_score = score(test_preds_ds[:, spark_ml_algo.prediction_feature])

            logger.info(f"Test score (#1 way): {test_score}")
            mlflow.log_metric("test_score", test_score)

        model_path = f"/tmp/models/spark-ml-pipe-lgb-light-{uid}"
        # 2. second way (Spark ML API, save-load-predict)
        with log_exec_timer("model_saving") as save_time:
            transformer = PipelineModel(stages=[sreader.transformer(add_array_attrs=True), ml_pipe.transformer()])
            transformer.write().overwrite().save(model_path)

            mlflow.log_param("model_path", model_path)

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
