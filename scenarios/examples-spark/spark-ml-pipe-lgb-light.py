import json
import logging.config
import logging.config
import os
import urllib.request
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
from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT, log_exec_time, log_exec_timer
from sparklightautoml.validation.iterators import SparkFoldsIterator

from examples_utils import get_persistence_manager, MLflowWrapperPersistenceManager
from examples_utils import get_spark_session, prepare_test_and_train, get_dataset_attrs

uid = uuid.uuid4()
log_filename = f'/tmp/slama-{uid}.log'
logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename=log_filename))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)

# NOTE! This demo requires datasets to be downloaded into a local folder.
# Run ./bin/download-datasets.sh to get required datasets into the folder.


def main(cv: int, seed: int, dataset_name: str = "lama_test_dataset"):
    spark = get_spark_session()

    mlflow.log_param("application_id", spark.sparkContext.applicationId)
    mlflow.log_param("executors", spark.conf.get("spark.executor.instances", None))
    mlflow.log_param("executor_cores", spark.conf.get("spark.executor.cores", None))
    mlflow.log_param("executor_memory", spark.conf.get("spark.executor.memory", None))
    mlflow.log_param("partitions_nums", spark.conf.get("spark.default.parallelism", None))
    mlflow.log_param("bucket_nums", os.environ.get("BUCKET_NUMS", None))
    mlflow.log_dict(dict(spark.sparkContext.getConf().getAll()), "spark_conf.json")

    spark.sparkContext.parallelize(list(range(10))).sum()

    exec_instances = int(spark.conf.get("spark.executor.instances", None))
    if exec_instances:
        url = f"{spark.sparkContext.uiWebUrl}/api/v1/applications/{spark.sparkContext.applicationId}/executors"
        with urllib.request.urlopen(url) as url:
            data = json.loads(url.read().decode())

        assert len(data) - 1 == exec_instances, \
            f"Incorrect number of executors. Expected: {exec_instances}. Found: {len(data) - 1}"

    path, task_type, roles, dtype = get_dataset_attrs(dataset_name)

    persistence_manager = MLflowWrapperPersistenceManager(get_persistence_manager())

    with log_exec_time():
        train_df, test_df = prepare_test_and_train(spark, path, seed)

        task = SparkTask(task_type)
        score = task.get_dataset_metric()

        sreader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False)
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

        mlflow.log_metric(fit_time.name, fit_time.duration)
        mlflow.log_metric("oof_score", oof_score)

        # 1. first way (LAMA API)
        with log_exec_timer("predict") as predict_time:
            test_sds = sreader.read(test_df, add_array_attrs=True)
            test_preds_ds = ml_pipe.predict(test_sds)
            test_score = score(test_preds_ds[:, spark_ml_algo.prediction_feature])

            logger.info(f"Test score (#1 way): {test_score}")

        mlflow.log_metric(predict_time.name, predict_time.duration)
        mlflow.log_metric("test_score", test_score)

        model_path = f"/tmp/models/spark-ml-pipe-lgb-light-{uid}"
        # 2. second way (Spark ML API, save-load-predict)
        with log_exec_timer("model_saving") as save_time:
            transformer = PipelineModel(stages=[sreader.transformer(add_array_attrs=True), ml_pipe.transformer()])
            transformer.write().overwrite().save(model_path)

        mlflow.log_metric(save_time.name, save_time.duration)
        mlflow.log_param("model_path", model_path)

        with log_exec_timer("model_loading") as load_time:
            pipeline_model = PipelineModel.load(model_path)

        mlflow.log_metric(load_time.name, load_time.duration)

        test_pred_df = pipeline_model.transform(test_df)
        test_pred_df = test_pred_df.select(
            SparkDataset.ID_COLUMN,
            F.col(roles['target']).alias('target'),
            F.col(spark_ml_algo.prediction_feature).alias('prediction')
        )
        test_score = score(test_pred_df)
        logger.info(f"Test score (#3 way): {test_score}")

    logger.info("Finished")

    spark.stop()

    log_files = bool(int(os.environ.get("LOG_FILES_TO_MLFLOW", "0")))
    if log_files:
        mlflow.log_artifact(log_filename, "run.log")


if __name__ == "__main__":
    log_to_mlflow = bool(int(os.environ.get("LOG_TO_MLFLOW", "0")))
    dataset_name = os.environ.get("DATASET", "lama_test_dataset")
    seed = int(os.environ.get("SEED", "42"))
    cv = int(os.environ.get("CV", "5"))

    if log_to_mlflow:
        exp_id = os.environ.get("EXPERIMENT", None)
        assert exp_id, "EXPERIMENT should be set if LOG_TO_MLFLOW is true"
        with mlflow.start_run(experiment_id=exp_id) as run:
            main(cv, seed, dataset_name)
    else:
        main(cv, seed, dataset_name)
