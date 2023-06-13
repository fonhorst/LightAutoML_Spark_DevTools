import datetime
import logging
import os
import uuid
from copy import deepcopy
from logging import config
from typing import Union, Callable, Optional, Dict

import mlflow
import optuna
from lightautoml.ml_algo.tuning.optuna import TunableAlgo
from lightautoml.ml_algo.utils import tune_and_fit_predict
from pyspark.sql import functions as sf
from sparklightautoml.computations.base import ComputationsSettings
from sparklightautoml.computations.utils import deecopy_tviter_without_dataset
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.ml_algo.base import SparkTabularMLAlgo
from sparklightautoml.ml_algo.tuning.parallel_optuna import ParallelOptunaTuner
from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT, log_exec_timer
from sparklightautoml.validation.base import SparkBaseTrainValidIterator
from sparklightautoml.validation.iterators import SparkFoldsIterator

from examples_utils import get_spark_session, train_test_split, ReportingParallelComputionsManager, \
    get_ml_algo, check_allocated_executors

logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename='/tmp/slama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


class ProgressReportingOptunaTuner(ParallelOptunaTuner):
    def __init__(self,
                 timeout: Optional[int] = 1000,
                 n_trials: Optional[int] = 100,
                 direction: Optional[str] = "maximize",
                 fit_on_holdout: bool = True,
                 random_state: int = 42,
                 parallelism: int = 1,
                 computations_manager: Optional[ComputationsSettings] = None,
                 stabilize: bool = False):
        super().__init__(timeout, n_trials, direction, fit_on_holdout, random_state, parallelism, computations_manager)
        self.run_reports: Dict[str, (float, float, dict)] = dict()
        self._stabilize = stabilize
        self.optimize_time: Optional[float] = None

    def _optimize(self, ml_algo: SparkTabularMLAlgo, train_valid_iterator: SparkBaseTrainValidIterator,
                  update_trial_time: Callable[[optuna.study.Study, optuna.trial.FrozenTrial], None]):
        with log_exec_timer("optimize_time") as timer:
            super()._optimize(ml_algo, train_valid_iterator, update_trial_time)
        self.optimize_time = timer.duration

    def _get_objective(self,
                       ml_algo: TunableAlgo,
                       estimated_n_trials: int,
                       train_valid_iterator: SparkBaseTrainValidIterator) \
            -> Callable[[optuna.trial.Trial], Union[float, int]]:
        assert isinstance(ml_algo, SparkTabularMLAlgo)

        def objective(trial: optuna.trial.Trial) -> float:
            start_time = datetime.datetime.now()
            trial_id = str(uuid.uuid4())
            with self._session.allocate() as slot:
                assert slot.dataset is not None
                _ml_algo = deepcopy(ml_algo)
                tv_iter = deecopy_tviter_without_dataset(train_valid_iterator)
                tv_iter.train = slot.dataset

                optimization_search_space = _ml_algo.optimization_search_space

                if not optimization_search_space:
                    optimization_search_space = _ml_algo._get_default_search_spaces(
                        suggested_params=_ml_algo.init_params_on_input(tv_iter),
                        estimated_n_trials=estimated_n_trials,
                    )

                if callable(optimization_search_space):
                     params = optimization_search_space(
                        trial=trial,
                        optimization_search_space=optimization_search_space,
                        suggested_params=_ml_algo.init_params_on_input(tv_iter),
                    )
                else:
                    params = self._sample(
                        trial=trial,
                        optimization_search_space=optimization_search_space,
                        suggested_params=_ml_algo.init_params_on_input(tv_iter),
                    )

                # we may intentionally do not use these params to stabilize execution time
                if not self._stabilize:
                    _ml_algo.params = params

                logger.warning(f"Optuna Params: {params}")

                with log_exec_timer(f"fit_{trial_id}") as fp_timer:
                    output_dataset = _ml_algo.fit_predict(train_valid_iterator=tv_iter)

                with log_exec_timer(f"scoring_{trial_id}") as scoring_timer:
                    obj_score = _ml_algo.score(output_dataset)

                end_time = datetime.datetime.now()

                date_fmt = "%Y-%m-%d %H:%M:%S.%f"
                self.run_reports[trial_id] = {
                    "start_time": start_time.strftime(date_fmt),
                    "end_time": end_time.strftime(date_fmt),
                    "start_time_ts": start_time.timestamp(),
                    "end_time_ts": end_time.timestamp(),
                    "fit_time": fp_timer.duration,
                    "scoring_time": scoring_timer.duration,
                    "score": obj_score,
                    "optuna_params": params,
                    "real_params": _ml_algo.params,
                    "stabilize": self._stabilize
                }

                logger.info(f"Objective score: {obj_score}")
                return obj_score

        return objective


if __name__ == "__main__":
    logger.info("In the very beginning of parallel-optuna")
    spark = get_spark_session()

    dataset_name = os.environ.get("DATASET", "lama_test_dataset")
    parallelism = int(os.environ.get("EXP_JOB_PARALLELISM", "1"))
    n_trials = 64
    timeout = 60000
    stabilize = False
    feat_pipe, default_params, ml_algo = get_ml_algo()
    dataset_path = f"file:///opt/spark_data/preproccessed_datasets/{dataset_name}__{feat_pipe}__features.dataset"

    # load and prepare data
    ds = SparkDataset.load(
        path=dataset_path,
        persistence_manager=PlainCachePersistenceManager()
    )

    check_allocated_executors()

    train_ds, test_ds = train_test_split(ds, test_slice_or_fold_num=4)

    # create main entities
    computations_manager_optuna = \
        ReportingParallelComputionsManager(parallelism=parallelism, use_location_prefs_mode=True)
    computations_manager_boosting = ReportingParallelComputionsManager(parallelism=1, use_location_prefs_mode=True)
    iterator = SparkFoldsIterator(train_ds).convert_to_holdout_iterator()
    with mlflow.start_run(experiment_id=os.environ["EXPERIMENT"]):
        tuner = ProgressReportingOptunaTuner(
            n_trials=n_trials,
            timeout=timeout,
            parallelism=parallelism,
            computations_manager=computations_manager_optuna,
            stabilize=stabilize
        )
        # tuner = DefaultTuner()

        # ml_algo = SparkBoostLGBM(
        #     default_params={"numIterations": 500, "earlyStoppingRound": 50_000},
        #     use_barrier_execution_mode=True,
        #     computations_settings=computations_manager_boosting
        # )
        ml_algo.computations_manager = computations_manager_boosting
        score = ds.task.get_dataset_metric()

        mlflow.log_params({
            "app_id": spark.sparkContext.applicationId,
            "app_name": spark.sparkContext.appName,
            "parallelism": parallelism,
            "dataset": dataset_name,
            "dataset_path": dataset_path,
            "feat_pipe": feat_pipe,
            "optuna_n_trials": n_trials,
            "optuna_timeout": timeout,
            "optuna_parallelism": parallelism,
            "optuna_stabilize": stabilize,
            "mlalgo_default_params": default_params
        })
        mlflow.log_dict(dict(spark.sparkContext.getConf().getAll()), "spark_conf.json")

        # fit and predict
        with log_exec_timer("ml_algo_time") as fit_timer:
            model, oof_preds = tune_and_fit_predict(ml_algo, tuner, iterator)

        mlflow.log_metric(fit_timer.name, fit_timer.duration)
        mlflow.log_metric("optuna_optimize_time", tuner.optimize_time)
        mlflow.log_metric(
            "optuna_prepare_dataset_pref_locs_time",
            computations_manager_optuna.last_session.prepare_dataset_time
        )
        mlflow.log_dict(tuner.run_reports, "run_reports.json")

        test_preds = ml_algo.predict(test_ds)

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
            test_metric_value = score(test_preds.data.select(
                SparkDataset.ID_COLUMN,
                sf.col(ds.target_column).alias('target'),
                sf.col(ml_algo.prediction_feature).alias('prediction')
            ))

        mlflow.log_metric(test_timer.name, test_timer.duration)
        mlflow.log_metric("test_metric_value", test_metric_value)

        print(f"OOF metric: {oof_metric_value}")
        print(f"Test metric: {oof_metric_value}")

    spark.stop()
