
import logging.config
import os

import mlflow
from importlib_metadata import version
from sparklightautoml.ml_algo.boost_lgbm import SparkBoostLGBM
from sparklightautoml.pipelines.features.lgb_pipeline import \
    SparkLGBSimpleFeatures
from sparklightautoml.pipelines.ml.base import SparkMLPipeline
from sparklightautoml.reader.base import SparkToSparkReader
from sparklightautoml.tasks.base import SparkTask as SparkTask
from sparklightautoml.utils import (VERBOSE_LOGGING_FORMAT, log_exec_timer,
                                    logging_config)
from sparklightautoml.validation.iterators import SparkHoldoutIterator

from examples_utils import (get_dataset_attrs, get_spark_session,
                            prepare_test_and_train)

logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename='/tmp/slama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)

# NOTE! This demo requires datasets to be downloaded into a local folder.
# Run ./bin/download-datasets.sh to get required datasets into the folder.


if __name__ == "__main__":
    spark = get_spark_session()

    seed = 42
    cv = int(os.getenv('CV'))
    dataset_name = os.getenv('DATASET_NAME')
    path, task_type, roles, dtype = get_dataset_attrs(dataset_name)

    ml_alg_kwargs = {
        'auto_unique_co': 10,
        'max_intersection_depth': 3,
        'multiclass_te_co': 3,
        'output_categories': True,
        'top_intersections': 4
    }
    cacher_key = "main_cache"

    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment('spark-ml-pipe-lgb-light.py')
    with mlflow.start_run():

        mlflow.log_param("pyspark", version('pyspark'))
        mlflow.log_param("synapseml", version('synapseml'))
        mlflow.log_param("spark.driver.cores", spark.sparkContext.getConf().get('spark.driver.cores'))
        mlflow.log_param("spark.driver.memory", spark.sparkContext.getConf().get('spark.driver.memory'))
        mlflow.log_param("spark.memory.fraction", spark.sparkContext.getConf().get('spark.memory.fraction'))
        mlflow.log_param("spark.executor.cores", spark.sparkContext.getConf().get('spark.executor.cores'))
        mlflow.log_param("spark.executor.memory", spark.sparkContext.getConf().get('spark.executor.memory'))
        mlflow.log_param("spark.executor.instances", spark.sparkContext.getConf().get('spark.executor.instances'))
        mlflow.log_param("spark.applicationId", spark.sparkContext.applicationId)

        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("cv", cv)
        mlflow.log_param("seed", seed)

        train_df, test_df = prepare_test_and_train(spark, path, seed)

        task = SparkTask(task_type)
        score = task.get_dataset_metric()

        sreader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False)
        sdataset = sreader.fit_read(train_df, roles=roles)

        iterator = SparkHoldoutIterator(sdataset)

        use_single_dataset_mode = os.getenv('USE_SINGLE_DATASET_MODE') == "True"
        mlflow.log_param("use_single_dataset_mode", use_single_dataset_mode)
        spark_ml_algo = SparkBoostLGBM(cacher_key=cacher_key, default_params={"earlyStoppingRound": 0}, freeze_defaults=False, use_single_dataset_mode=use_single_dataset_mode)
        spark_features_pipeline = SparkLGBSimpleFeatures(cacher_key=cacher_key)

        ml_pipe = SparkMLPipeline(
            cacher_key=cacher_key,
            ml_algos=[spark_ml_algo],
            pre_selection=None,
            features_pipeline=spark_features_pipeline,
            post_selection=None
        )

        with log_exec_timer("ml_pipe.fit_predict()") as train_timer:
            oof_preds_ds = ml_pipe.fit_predict(iterator)

        mlflow.log_metric("ml_pipe.fit_predict_secs", train_timer.duration)

        oof_score = score(oof_preds_ds[:, spark_ml_algo.prediction_feature])
        mlflow.log_metric("oof_score", oof_score)
        logger.info(f"OOF score: {oof_score}")

        with log_exec_timer("spark-lama predicting on test") as predict_timer:
            # 1. first way (LAMA API)
            test_sds = sreader.read(test_df, add_array_attrs=True)
            test_preds_ds = ml_pipe.predict(test_sds)
            test_score = score(test_preds_ds[:, spark_ml_algo.prediction_feature])
            logger.info(f"Test score: {test_score}")

        mlflow.log_metric("test_score", test_score)
        mlflow.log_metric("predict_duration_secs", predict_timer.duration)
        

    logger.info("Finished")

    spark.stop()
