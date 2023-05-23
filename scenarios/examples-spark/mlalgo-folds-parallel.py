import logging
from logging import config
from typing import Tuple, Union

import os

import mlflow
from lightautoml.ml_algo.tuning.base import DefaultTuner
from lightautoml.ml_algo.utils import tune_and_fit_predict
from pyspark.sql import functions as sf, SparkSession
from sparklightautoml.computations.manager import ParallelComputationsManager, get_executors, get_executors_cores
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.ml_algo.boost_lgbm import SparkBoostLGBM
from sparklightautoml.ml_algo.linear_pyspark import SparkLinearLBFGS
from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT, log_exec_timer
from sparklightautoml.validation.iterators import SparkFoldsIterator
from examples_utils import get_spark_session, initialize_environment

logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename='/tmp/slama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


def train_test_split(dataset: SparkDataset, test_slice_or_fold_num: Union[float, int] = 0.2) \
        -> Tuple[SparkDataset, SparkDataset]:

    exec_count = len(get_executors())
    exec_cores = get_executors_cores()

    if isinstance(test_slice_or_fold_num, float):
        assert 0 <= test_slice_or_fold_num <= 1
        train, test = dataset.data.randomSplit([1 - test_slice_or_fold_num, test_slice_or_fold_num])
    else:
        train = dataset.data.where(sf.col(dataset.folds_column) != test_slice_or_fold_num)\
            .repartition(exec_count * exec_cores)
        test = dataset.data.where(sf.col(dataset.folds_column) == test_slice_or_fold_num)\
            .repartition(exec_count * exec_cores)

    train_dataset, test_dataset = dataset.empty(), dataset.empty()
    train_dataset.set_data(train, dataset.features, roles=dataset.roles)
    test_dataset.set_data(test, dataset.features, roles=dataset.roles)

    return train_dataset, test_dataset


@initialize_environment
def main(spark: SparkSession):
    feat_pipe = "linear"  # linear, lgb_simple or lgb_adv
    ml_algo_name = "linear_l2"  # linear_l2, lgb
    job_parallelism = int(os.environ.get("EXP_JOB_PARALLELISM", "1"))
    dataset_name = os.environ.get("DATASET", "lama_test_dataset")
    dataset_path = f"file:///opt/spark_data/preproccessed_datasets/{dataset_name}__{feat_pipe}__features.dataset"

    mlflow.log_params({
        "feat_pipe": feat_pipe,
        "ml_algo": ml_algo_name,
        "dataset": dataset_name,
        "job_parallelism": job_parallelism,
        "dataset_path": dataset_path
    })

    logger.info(f"Job Parallelism: {job_parallelism}")

    # load and prepare data
    ds = SparkDataset.load(
        path=dataset_path,
        persistence_manager=PlainCachePersistenceManager()
    )
    train_ds, test_ds = train_test_split(ds, test_slice_or_fold_num=4)
    train_ds, test_ds = train_ds.persist(), test_ds.persist()

    # create main entities
    computations_manager = ParallelComputationsManager(job_pool_size=job_parallelism)
    iterator = SparkFoldsIterator(train_ds)  # .convert_to_holdout_iterator()
    if ml_algo_name == "lgb":
        ml_algo = SparkBoostLGBM(experimental_parallel_mode=True, computations_manager=computations_manager)
    else:
        ml_algo = SparkLinearLBFGS(default_params={'regParam': [1e-5]}, computations_manager=computations_manager)

    score = ds.task.get_dataset_metric()

    # fit and predict
    with log_exec_timer("fit") as timer:
        model, oof_preds = tune_and_fit_predict(ml_algo, DefaultTuner(), iterator)

    mlflow.log_metric(timer.name, timer.duration)

    with log_exec_timer("predict") as timer:
        test_preds = model.predict(test_ds)

    mlflow.log_metric(timer.name, timer.duration)

    # estimate oof and test metrics
    oof_metric_value = score(oof_preds.data.select(
        SparkDataset.ID_COLUMN,
        sf.col(ds.target_column).alias('target'),
        sf.col(ml_algo.prediction_feature).alias('prediction')
    ))

    test_metric_value = score(test_preds.data.select(
        SparkDataset.ID_COLUMN,
        sf.col(ds.target_column).alias('target'),
        sf.col(ml_algo.prediction_feature).alias('prediction')
    ))

    mlflow.log_metric("OOF_score", oof_metric_value)
    mlflow.log_metric("Test_score", test_metric_value)

    logger.info(f"OOF metric: {oof_metric_value}")
    logger.info(f"Test metric: {test_metric_value}")


if __name__ == "__main__":
    main()
