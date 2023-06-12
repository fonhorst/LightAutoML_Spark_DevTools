import os

import mlflow
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.ml_algo.boost_lgbm import SparkBoostLGBM
from sparklightautoml.ml_algo.linear_pyspark import SparkLinearLBFGS
from sparklightautoml.utils import log_exec_timer
from sparklightautoml.validation.iterators import SparkFoldsIterator
from pyspark.sql import functions as sf

from examples_utils import get_spark_session, train_test_split


def get_ml_algo():
    ml_algo_name = os.environ.get("EXP_ML_ALGO", "linear_l2")

    if ml_algo_name == "linear_l2":
        feat_pipe = "linear"  # linear, lgb_simple or lgb_adv
        default_params = {'regParam': [1e-5], "maxIter": 100, "aggregationDepth": 2, "tol": 0.0}
        ml_algo = SparkLinearLBFGS(default_params)
    elif ml_algo_name == "lgb":
        feat_pipe = "lgb_adv"  # linear, lgb_simple or lgb_adv
        default_params = {"numIterations": 500, "earlyStoppingRound": 50_000}
        ml_algo = SparkBoostLGBM(default_params, use_barrier_execution_mode=True)
    else:
        raise ValueError(f"Unknown ml algo: {ml_algo_name}")

    return feat_pipe, default_params, ml_algo


def main():
    spark = get_spark_session()

    feat_pipe, default_params, ml_algo = get_ml_algo()
    dataset_name = os.environ.get("DATASET", "lama_test_dataset")
    dataset_path = f"file:///opt/spark_data/preproccessed_datasets/{dataset_name}__{feat_pipe}__features.dataset"

    # load and prepare data
    ds = SparkDataset.load(
        path=dataset_path,
        persistence_manager=PlainCachePersistenceManager()
    )

    score = ds.task.get_dataset_metric()
    train_ds, test_ds = train_test_split(ds, test_slice_or_fold_num=4)
    iterator = SparkFoldsIterator(train_ds).convert_to_holdout_iterator()

    with mlflow.start_run(experiment_id=os.environ["EXPERIMENT"]):
        mlflow.log_params({
            "app_id": spark.sparkContext.applicationId,
            "app_name": spark.sparkContext.appName,
            "dataset": dataset_name,
            "dataset_path": dataset_path,
            "feat_pipe": feat_pipe,
            "mlalgo_default_params": default_params
        })
        mlflow.log_dict(dict(spark.sparkContext.getConf().getAll()), "spark_conf.json")

        with log_exec_timer("ml_algo_time") as fit_timer:
            oof_preds = ml_algo.fit_predict(iterator)

        mlflow.log_metric(fit_timer.name, fit_timer.duration)

        with log_exec_timer("oof_score_time") as oof_timer:
            # estimate oof and test metrics
            oof_metric_value = score(oof_preds.data.select(
                SparkDataset.ID_COLUMN,
                sf.col(ds.target_column).alias('target'),
                sf.col(ml_algo.prediction_feature).alias('prediction')
            ))

        mlflow.log_metric(oof_timer.name, oof_timer.duration)
        mlflow.log_metric("oof_metric_value", oof_metric_value)

        with log_exec_timer("test_score_time") as test_timer:
            test_preds = ml_algo.predict(test_ds)
            test_metric_value = score(test_preds.data.select(
                SparkDataset.ID_COLUMN,
                sf.col(ds.target_column).alias('target'),
                sf.col(ml_algo.prediction_feature).alias('prediction')
            ))

        mlflow.log_metric(test_timer.name, test_timer.duration)
        mlflow.log_metric("test_metric_value", test_metric_value)

        print(f"OOF metric: {oof_metric_value}")
        print(f"Test metric: {oof_metric_value}")


if __name__ == "__main__":
    main()
