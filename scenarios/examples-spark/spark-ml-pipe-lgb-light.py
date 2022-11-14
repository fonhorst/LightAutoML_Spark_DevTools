import logging.config
import logging.config
import os

import mlflow
from pyspark.ml import PipelineModel
from pyspark.sql import functions as F

from examples_utils import get_persistence_manager
from examples_utils import get_spark_session, prepare_test_and_train, get_dataset_attrs
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.ml_algo.boost_lgbm import SparkBoostLGBM
from sparklightautoml.pipelines.features.lgb_pipeline import SparkLGBSimpleFeatures
from sparklightautoml.pipelines.ml.base import SparkMLPipeline
from sparklightautoml.reader.base import SparkToSparkReader
from sparklightautoml.tasks.base import SparkTask as SparkTask
from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT, log_exec_time, log_exec_timer
from sparklightautoml.validation.iterators import SparkFoldsIterator

logging.config.dictConfig(logging_config(level=logging.INFO, log_filename='/tmp/slama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)

# NOTE! This demo requires datasets to be downloaded into a local folder.
# Run ./bin/download-datasets.sh to get required datasets into the folder.


def main():
    with open("/tmp/slama.log") as f:
        log_content = f.read()
    mlflow.log_text(log_content, "slama.log")

# def main():
#     spark = get_spark_session()
#
#     seed = 42
#     cv = 5
#     dataset_name = "lama_test_dataset"
#     path, task_type, roles, dtype = get_dataset_attrs(dataset_name)
#
#     persistence_manager = get_persistence_manager()
#
#     with log_exec_time():
#         train_df, test_df = prepare_test_and_train(spark, path, seed)
#
#         task = SparkTask(task_type)
#         score = task.get_dataset_metric()
#
#         sreader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False)
#         sdataset = sreader.fit_read(train_df, roles=roles, persistence_manager=persistence_manager)
#
#         iterator = SparkFoldsIterator(sdataset).convert_to_holdout_iterator()
#
#         spark_ml_algo = SparkBoostLGBM(freeze_defaults=False, use_single_dataset_mode=True)
#         spark_features_pipeline = SparkLGBSimpleFeatures()
#
#         ml_pipe = SparkMLPipeline(
#             ml_algos=[spark_ml_algo],
#             pre_selection=None,
#             features_pipeline=spark_features_pipeline,
#             post_selection=None
#         )
#
#         with log_exec_timer("fit_time") as fit_time:
#             oof_preds_ds = ml_pipe.fit_predict(iterator)
#             oof_score = score(oof_preds_ds[:, spark_ml_algo.prediction_feature])
#
#             logger.info(f"OOF score: {oof_score}")
#
#         mlflow.log_param(fit_time.name, fit_time.duration)
#         mlflow.log_param("oof_score", oof_score)
#
#         # 1. first way (LAMA API)
#         with log_exec_timer("predict") as predict_time:
#             test_sds = sreader.read(test_df, add_array_attrs=True)
#             test_preds_ds = ml_pipe.predict(test_sds)
#             test_score = score(test_preds_ds[:, spark_ml_algo.prediction_feature])
#
#             logger.info(f"Test score (#1 way): {test_score}")
#
#         mlflow.log_param(predict_time.name, predict_time.duration)
#         mlflow.log_param("test_score", test_score)
#
#         # 2. second way (Spark ML API, save-load-predict)
#         with log_exec_timer("model_saving") as save_time:
#             transformer = PipelineModel(stages=[sreader.transformer(add_array_attrs=True), ml_pipe.transformer()])
#             transformer.write().overwrite().save("/tmp/reader_and_spark_ml_pipe_lgb")
#
#         mlflow.log_param(save_time.name, save_time.duration)
#
#         with log_exec_timer("model_loading") as load_time:
#             pipeline_model = PipelineModel.load("/tmp/reader_and_spark_ml_pipe_lgb")
#
#         mlflow.log_param(load_time.name, load_time.duration)
#
#         test_pred_df = pipeline_model.transform(test_df)
#         test_pred_df = test_pred_df.select(
#             SparkDataset.ID_COLUMN,
#             F.col(roles['target']).alias('target'),
#             F.col(spark_ml_algo.prediction_feature).alias('prediction')
#         )
#         test_score = score(test_pred_df)
#         logger.info(f"Test score (#3 way): {test_score}")
#
#     logger.info("Finished")
#
#     spark.stop()


if __name__ == "__main__":
    log_to_mlflow = bool(int(os.environ.get("LOG_TO_MLFLOW", "0")))

    if log_to_mlflow:
        exp_id = os.environ.get("EXPERIMENT", None)
        assert exp_id, "EXPERIMENT should be set if LOG_TO_MLFLOW is true"
        with mlflow.start_run(experiment_id=exp_id) as run:
            main()
    else:
        main()
