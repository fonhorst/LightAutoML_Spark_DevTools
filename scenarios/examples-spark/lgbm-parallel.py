import functools
import logging
import math
import os
import threading
import warnings
from contextlib import contextmanager
from copy import deepcopy
from enum import Enum
from multiprocessing.pool import ThreadPool
from typing import Tuple, List, Optional

# noinspection PyUnresolvedReferences
from attr import dataclass
from pyspark import inheritable_thread_target
from pyspark import keyword_only
from pyspark.ml.common import inherit_doc
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.wrapper import JavaTransformer
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf
from sparklightautoml.computations.manager import get_executors, get_executors_cores
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager
from sparklightautoml.utils import log_exec_timer, JobGroup
from synapse.ml.lightgbm import LightGBMClassifier, LightGBMRegressor

from examples_utils import get_spark_session

logger = logging.getLogger(__name__)


base_params = {
    'learningRate': 0.01,
    'numLeaves': 32,
    'featureFraction': 0.7,
    'baggingFraction': 0.7,
    'baggingFreq': 1,
    'maxDepth': -1,
    'minGainToSplit': 0.0,
    'maxBin': 255,
    'minDataInLeaf': 5,
    'numIterations': 3000,
    'earlyStoppingRound': 200,
    'objective': 'binary',
    'metric': 'auc'
}

# Problems:
# 1. Third-party concurrent workload may occupy cores on desired locations,
#   that will lead to not completely equal amount of computational resources to lgbm workers
#
# 2. Other workload may occupy cores used by lgbm threads
#   (because lightgbm letted them go free while still calculating with the main task)
#
# 3. Other workload may be harmed by cores overcommit due to incorrect threads num when not enough cores
#   allocated per executor for an lgbm worker (for instance, we asked for 3 cores per exec in an app
#   with 6 cores executors and ends up wuth allocating only 1 core per some executor,
#   but we will still be using 3 threads. The vice versa is also possible: 3 threads is in the settings,
#   while we ends up with 5 cores allocated per some executor)
#
# 4. Running lgbm workload without barrier may lead to softlock
#
# Note: p.1 (and p.2, 3 too) can be alleviated by controlling what can be executed in parallel
# by forcing no other parallel workload when parallel lightgbms is submitted for execution.
# (What to do with treeAggregating in parallel instance of lightgbm,
# that can potentially ruin allocations of next instances in the queue?
# if it is just test, than send them trough the same PrefferedLocsTransformer).
# Also, preparation step that moves and caches data may be required to get rid of this problem.
# We intentialy pre-move dataset copies and caches onto nodes in the correspondence
# with future instances allocation and only after that we start computations.
# (It is actual for pre-start action of lightgbm)
# There is a one alternative to the pre-moving. We may set 'shuffle=false' for the ds.rdd.coalesce(...)
# and than redefine the logic of 'PrefferedLocsPartitionCoalescer' to combine partitions by ourselves in order
# to avoid shuffle stage that forces to add additional tasks for data redistributing (need to test this hypothesis).
#
# Note: in case of failures or switching number of executors,
# The performance may be decreased due to unaligned allocation of workers, but still will be continued.
# Potential solution here may be based on constant monitoring in changes of spark app executors
# and realigning allocations on the fly.
#
# Note: p.2 and 3 may be solved by tweaking internals of lgbm itself
# (using wrapper that can redefine its behaviour in such situations).
# Also, it can be at least partially controlled by external service that performs workload alignment
# and prevent overcommitting "manually".
#
#
# num_executors = <defined by app>
# num_cores_per_executor = <defined by app>
# all_cores = num_executors * num_cores_per_executor
#
# max_parallelism = <defined by user> and <upper limit capped by other settings>
#
# Parallel modes for lgbm:
#   1. Simple
#       UseSingleDatasetMode=False, num_tasks=math.ceil(all_cores / max_parallelism), barrier=True, num_threads=1
#
#   2. One task per executor
#       UseSingleDatasetMode=True, num_tasks=num_executors, barrier=True, num_threads=1
#       max_parallelism == num_cores_per_executor
#
#       Cons: can be harmed by p.1, 3
#       Pros: better performance than 1; NO need to move data
#
#   3. Several tasks per executor
#       UseSingleDatasetMode=True, num_tasks=num_executors * num_cores, barrier=True, num_threads=num_cores
#       num_cores = <defined by user>
#       max_parallelism == math.floor(num_cores_per_executor / num_cores)
#
#       Cons: can be harmed by p.1, 2, 3
#       Pros: potentially better performance than 1, 2, 4; NO need to move data
#
#   4. One task per subset of executors
#       UseSingleDatasetMode=True, num_tasks=custom_num_tasks, barrier=True, num_threads=1
#       custom_num_tasks = <defined by user>, custom_num_tasks < num_executors
#       max_parallelism = math.floor(all_cores / custom_num_tasks)
#
#       Cons: can be harmed by p.1, 3; need to move data
#       Pros: better performance than 1
#
#   5. One lgbm instance per executors subset
#       UseSingleDatasetMode=True, num_tasks=num_execs_per_instance * num_cores_per_executor,
#           barrier=True, num_threads=num_cores_per_executor
#       num_execs_per_instance = <defined by user>
#       max_parallelism = math.floor(num_executors / num_execs_per_instance)
#
#       Cons: can be harmed by p.1, 2, 3; need to move data
#       Pros: potentially better performance than 1, 2, 3, 4


@inherit_doc
class PrefferedLocsPartitionCoalescerTransformer(JavaTransformer):
    """
    Custom implementation of PySpark BalancedUnionPartitionsCoalescerTransformer wrapper
    """

    @keyword_only
    def __init__(self, pref_locs: List[str], do_shuffle: bool = True):
        super(PrefferedLocsPartitionCoalescerTransformer, self).__init__()
        self._java_obj = self._new_java_obj(
            "org.apache.spark.lightautoml.utils.PrefferedLocsPartitionCoalescerTransformer",
            self.uid, pref_locs, do_shuffle
        )


@dataclass
class DatasetSlot:
    train_df: DataFrame
    test_df: DataFrame
    pref_locs: Optional[List[str]]
    num_tasks: int
    num_threads: int
    use_single_dataset_mode: bool
    free: bool
    id: int


class ParallelismMode(Enum):
    pref_locs = 1
    no_single_dataset_mode = 2
    single_dataset_mode = 3
    no_parallelism = 4


class ParallelExperiment:
    def __init__(self,
                 spark: SparkSession,
                 dataset_name: str,
                 parallelism_mode: ParallelismMode = ParallelismMode.pref_locs):
        self.spark = spark
        self.parallelism_mode = parallelism_mode
        self.base_dataset_path = f"/opt/spark_data/parallel_slama_{dataset_name}"
        self.train_path = os.path.join(self.base_dataset_path, "train.parquet")
        self.test_path = os.path.join(self.base_dataset_path, "test.parquet")
        self.metadata_path = os.path.join(self.base_dataset_path, "metadata.parquet")

        self._executors = get_executors()
        self._cores_per_exec = get_executors_cores()

        # for ParallelismMode.pref_locs
        self._train_slots: Optional[List[DatasetSlot]] = None
        self._train_slots_lock = threading.Lock()

        # for ParallelismMode.no_single_dataset_mode
        self._slot: Optional[DatasetSlot] = None

    def prepare_trains(self, dataset: SparkDataset, max_job_parallelism: int):
        train_df = dataset.data
        test_df = dataset.data

        if self.parallelism_mode == ParallelismMode.pref_locs:
            execs_per_job = max(1, math.floor(len(self._executors) / max_job_parallelism))

            if len(self._executors) % max_job_parallelism != 0:
                warnings.warn(f"Uneven number of executors per job. Setting execs per job: {execs_per_job}.")

            slots_num = int(len(self._executors) / execs_per_job)

            def _coalesce_df_to_locs(df: DataFrame, pref_locations: List[str]):
                df = PrefferedLocsPartitionCoalescerTransformer(pref_locs=pref_locations).transform(df)
                df = df.cache()
                df.write.mode('overwrite').format('noop').save()
                return df

            # path = "/tmp/train_df_for_coalescing.parquet"
            # train_df.write.parquet(path)
            # spark = SparkSession.getActiveSession()

            def build_task(slot_num: int):
                def func():
                    pref_locs = self._executors[slot_num * execs_per_job: (slot_num + 1) * execs_per_job]

                    # df = spark.read.parquet(path)

                    # prepare train
                    tr_df = _coalesce_df_to_locs(train_df, pref_locs)

                    # # prepare test
                    # test_df = _coalesce_df_to_locs(test_df, pref_locs)
                    tst_df = tr_df

                    slot = DatasetSlot(
                        train_df=tr_df,
                        test_df=tst_df,
                        pref_locs=pref_locs,
                        num_tasks=len(pref_locs) * self._cores_per_exec,
                        num_threads=-1,
                        use_single_dataset_mode=True,
                        free=True,
                        id=slot_num
                    )

                    logger.info(f"Pref lcos for slot #{slot_num}: {pref_locs}")
                    return slot
                return func

            pool = ThreadPool(processes=slots_num)
            tasks = map(inheritable_thread_target, [build_task(i) for i in range(slots_num)])
            self._train_slots = list(pool.imap_unordered(lambda f: f(), tasks))
        elif self.parallelism_mode == ParallelismMode.no_single_dataset_mode:
            num_tasks_per_job = max(1, math.floor(len(self._executors) * self._cores_per_exec / max_job_parallelism))
            self._slot = DatasetSlot(
                train_df=train_df,
                test_df=test_df,
                pref_locs=None,
                num_tasks=num_tasks_per_job,
                num_threads=-1,
                use_single_dataset_mode=False,
                free=False,
                id=0
            )
        elif self.parallelism_mode == ParallelismMode.single_dataset_mode:
            num_tasks_per_job = max(1, math.floor(len(self._executors) * self._cores_per_exec / max_job_parallelism))
            num_threads_per_exec = max(1, math.floor(num_tasks_per_job / len(self._executors)))

            if num_threads_per_exec != 1:
                warnings.warn(f"Num threads per exec {num_threads_per_exec} != 1. "
                              f"Overcommitting or undercommiting may happen due to "
                              f"uneven allocations of cores between executors for a job")

            self._slot = DatasetSlot(
                train_df=train_df,
                test_df=test_df,
                pref_locs=None,
                num_tasks=num_tasks_per_job,
                num_threads=num_threads_per_exec,
                use_single_dataset_mode=True,
                free=False,
                id=-1
            )

        else:
            self._slot = DatasetSlot(
                train_df=train_df,
                test_df=test_df,
                pref_locs=None,
                num_tasks=train_df.rdd.getNumPartitions(),
                num_threads=-1,
                use_single_dataset_mode=True,
                free=False,
                id=-1
            )

    @contextmanager
    def _use_train(self) -> DatasetSlot:
        if self.parallelism_mode == ParallelismMode.pref_locs:
            with self._train_slots_lock:
                free_slot = next((slot for slot in self._train_slots if slot.free))
                free_slot.free = False

            yield free_slot

            with self._train_slots_lock:
                free_slot.free = True
        else:
            yield self._slot

    def train_model(self, dataset: SparkDataset, run_num: int) -> Tuple[int, float]:
        logger.info(f"Starting to train the model for run #{run_num}")

        task_type = dataset.task.name
        roles = dataset.roles
        target = dataset.target_column
        spark = dataset.data.sql_ctx.sparkSession

        prediction_col = 'LightGBM_prediction_0'
        params = deepcopy(base_params)
        if task_type in ["binary", "multiclass"]:
            params["rawPredictionCol"] = 'raw_prediction'
            params["probabilityCol"] = prediction_col
            params["predictionCol"] = 'prediction'
            params["isUnbalance"] = True
        else:
            params["predictionCol"] = prediction_col

        if task_type == "reg":
            params["objective"] = "regression"
            params["metric"] = "mse"
        elif task_type == "binary":
            params["objective"] = "binary"
            params["metric"] = "auc"
        elif task_type == "multiclass":
            params["objective"] = "multiclass"
            params["metric"] = "multiclass"
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        if task_type != "reg":
            if "alpha" in params:
                del params["alpha"]
            if "lambdaL1" in params:
                del params["lambdaL1"]
            if "lambdaL2" in params:
                del params["lambdaL2"]

        assembler = VectorAssembler(
            inputCols=list(roles.keys()),
            outputCol=f"LightGBM_vassembler_features",
            handleInvalid="keep"
        )

        with self._use_train() as slot:
            slot: DatasetSlot = slot
            train_df, test_df, num_tasks, num_threads, use_single_dataset \
                = slot.train_df, slot.test_df, slot.num_tasks, slot.num_threads, slot.use_single_dataset_mode
            full_data = train_df.withColumn('is_val', sf.col('reader_fold_num') == 0)

            lgbm_booster = LightGBMRegressor if task_type == "reg" else LightGBMClassifier

            if num_threads != -1:
                params['numThreads'] = num_threads

            params["numIterations"] = 500

            lgbm = lgbm_booster(
                **params,
                featuresCol=assembler.getOutputCol(),
                labelCol=target,
                # validationIndicatorCol='is_val',
                verbosity=1,
                useSingleDatasetMode=use_single_dataset,
                isProvideTrainingMetric=True,
                chunkSize=4_000_000,
                useBarrierExecutionMode=True,
                numTasks=num_tasks
            )

            if task_type == "reg":
                lgbm.setAlpha(0.5).setLambdaL1(0.0).setLambdaL2(0.0)

            with JobGroup(f"Run {run_num} in slot #{slot.id}", f"Should be executed on {slot.pref_locs}", spark):
                transformer = lgbm.fit(assembler.transform(full_data))

            # with JobGroup(f"Run {run_num} in slot #{slot.id}", f"Scoring. "
            #                                                    f"Should be executed on {slot.pref_locs}", spark):
            #     # metric_value = -1.0
            #     preds_df = transformer.transform(assembler.transform(test_df))
            #     score = SparkTask(task_type).get_dataset_metric()
            #     metric_value = score(
            #         preds_df.select(
            #             SparkDataset.ID_COLUMN,
            #             sf.col(target).alias('target'),
            #             sf.col(prediction_col).alias('prediction')
            #         )
            #     )
            metric_value = 0.0

            logger.info(f"Finished training the model for run #{run_num}")

        return run_num, metric_value

    def run(self, dataset: SparkDataset, max_job_parallelism: int = 3, repeatitions: int = 10) \
            -> List[Tuple[int, float]]:
        with log_exec_timer("Parallel experiment runtime"):
            logger.info("Starting to run the experiment")

            self.prepare_trains(dataset, max_job_parallelism)

            tasks = [
                functools.partial(self.train_model, dataset, i)
                for i in range(repeatitions)
            ]

            processes_num = 1 if self.parallelism_mode == ParallelismMode.no_parallelism else max_job_parallelism
            pool = ThreadPool(processes=processes_num)
            tasks = map(inheritable_thread_target, tasks)
            results = (result for result in pool.imap_unordered(lambda f: f(), tasks) if result)
            results = sorted(results, key=lambda x: x[0])

            logger.info("The experiment is finished")
            return results


def main():
    spark = get_spark_session()
    feat_pipe = "lgb_adv"
    dataset_name = os.environ.get("DATASET", "lama_test_dataset")
    dataset_path = f"file:///opt/spark_data/preproccessed_datasets/{dataset_name}__{feat_pipe}__features.dataset"

    # load and prepare data
    ds = SparkDataset.load(
        path=dataset_path,
        persistence_manager=PlainCachePersistenceManager(),
        # partitions_num=len(get_executors()) * get_executors_cores()
    )

    exp = ParallelExperiment(
        spark,
        dataset_name=os.environ.get("DATASET", "used_cars_dataset"),
        parallelism_mode=ParallelismMode[os.environ.get("PARALLELISM_MODE", "pref_locs")]
    )

    results = exp.run(ds, max_job_parallelism=int(os.environ.get("EXP_JOB_PARALLELISM", "3")), repeatitions=8)

    for run_num, metric_value in results:
        logger.info(f"Metric value (run_num = {run_num}: {metric_value}")

    spark.stop()


if __name__ == "__main__":
    main()
