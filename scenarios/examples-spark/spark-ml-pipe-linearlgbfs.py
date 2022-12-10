import logging.config
import logging.config
import os
import uuid

import mlflow
from lightautoml.pipelines.selection.importance_based import ImportanceCutoffSelector, ModelBasedImportanceEstimator
from sparklightautoml.ml_algo.boost_lgbm import SparkBoostLGBM
from sparklightautoml.ml_algo.linear_pyspark import SparkLinearLBFGS
from sparklightautoml.pipelines.features.lgb_pipeline import SparkLGBSimpleFeatures
from sparklightautoml.pipelines.features.linear_pipeline import SparkLinearFeatures
from sparklightautoml.pipelines.ml.base import SparkMLPipeline
from sparklightautoml.pipelines.selection.base import BugFixSelectionPipelineWrapper
from sparklightautoml.pipelines.selection.base import SparkSelectionPipelineWrapper
from sparklightautoml.reader.base import SparkToSparkReader
from sparklightautoml.tasks.base import SparkTask as SparkTask
from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT, log_exec_timer
from sparklightautoml.validation.iterators import SparkFoldsIterator

from examples_utils import get_persistence_manager, mlflow_deco, log_session_params_to_mlflow
from examples_utils import get_spark_session, get_dataset_attrs, prepare_test_and_train

uid = uuid.uuid4()
log_filename = f'/tmp/slama-{uid}.log'
logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename=log_filename))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)

# NOTE! This demo requires datasets to be downloaded into a local folder.
# Run ./bin/download-datasets.sh to get required datasets into the folder.


@mlflow_deco
def main(cv: int = 5, seed: int = 42, dataset_name: str = "lama_test_dataset"):
    parallelism = int(os.environ.get("SLAMA_PARALLELISM", "1"))
    spark = get_spark_session()
    persistence_manager = get_persistence_manager(run_id=str(uid))
    path, task_type, roles, dtype = get_dataset_attrs(dataset_name)

    ml_alg_kwargs = {
        'auto_unique_co': 10,
        'max_intersection_depth': 3,
        'multiclass_te_co': 3,
        'output_categories': True,
        'top_intersections': 4
    }

    log_session_params_to_mlflow()
    mlflow.log_param("seed", seed)
    mlflow.log_param("cv", cv)
    mlflow.log_param("dataset", dataset_name)
    mlflow.log_param("parallelism", parallelism)

    with log_exec_timer("full_time") as full_timer:
        train_df, test_df = prepare_test_and_train(spark, path, seed)

        task = SparkTask(task_type)
        score = task.get_dataset_metric()

        sreader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False)
        sdataset = sreader.fit_read(train_df, roles=roles, persistence_manager=persistence_manager)

        iterator = SparkFoldsIterator(sdataset, n_folds=cv)

        spark_ml_algo = SparkLinearLBFGS(default_params={'regParam': [1e-5]}, parallelism=parallelism)
        spark_features_pipeline = SparkLinearFeatures(**ml_alg_kwargs)
        spark_selector = BugFixSelectionPipelineWrapper(ImportanceCutoffSelector(
            cutoff=0.0,
            feature_pipeline=SparkLGBSimpleFeatures(),
            ml_algo=SparkBoostLGBM(freeze_defaults=False),
            imp_estimator=ModelBasedImportanceEstimator()
        ))

        ml_pipe = SparkMLPipeline(
            ml_algos=[spark_ml_algo],
            pre_selection=SparkSelectionPipelineWrapper(spark_selector),
            features_pipeline=spark_features_pipeline,
            post_selection=None
        )

        with log_exec_timer("fit_time") as timer:
            oof_preds_ds = ml_pipe.fit_predict(iterator).persist()
            oof_score = score(oof_preds_ds[:, spark_ml_algo.prediction_feature])
            logger.info(f"OOF score: {oof_score}")

        mlflow.log_metric("oof_score", oof_score)
        mlflow.log_metric(timer.name, timer.duration)

        # 1. first way (LAMA API)
        with log_exec_timer("predict_time") as timer:
            test_sds = sreader.read(test_df, add_array_attrs=True)
            test_preds_ds = ml_pipe.predict(test_sds)
            test_score = score(test_preds_ds[:, spark_ml_algo.prediction_feature])
            logger.info(f"Test score (#1 way): {test_score}")

        mlflow.log_metric("test_score", oof_score)
        mlflow.log_metric(timer.name, timer.duration)

        # # 2. second way (Spark ML API)
        # transformer = PipelineModel(stages=[sreader.transformer(add_array_attrs=True), ml_pipe.transformer()])
        # test_pred_df = transformer.transform(test_df)
        # test_pred_df = test_pred_df.select(
        #     SparkDataset.ID_COLUMN,
        #     sf.col(roles['target']).alias('target'),
        #     sf.col(spark_ml_algo.prediction_feature).alias('prediction')
        # )
        # test_score = score(test_pred_df)
        # logger.info(f"Test score (#2 way): {test_score}")

    logger.info("Finished")

    mlflow.log_metric(full_timer.name, full_timer.duration)

    oof_preds_ds.unpersist()
    # this is necessary if persistence_manager is of CompositeManager type
    # it may not be possible to obtain oof_predictions (predictions from fit_predict) after calling unpersist_all
    persistence_manager.unpersist_all()

    log_files = bool(int(os.environ.get("LOG_FILES_TO_MLFLOW", "0")))
    if log_files:
        for handler in logger.handlers:
            handler.flush()
        mlflow.log_artifact(log_filename, "run.log")

    spark.stop()


if __name__ == "__main__":
    main()
