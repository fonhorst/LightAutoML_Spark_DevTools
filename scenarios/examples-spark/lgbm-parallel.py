import functools
import logging
import os
import pickle
import threading
import warnings
from contextlib import contextmanager
from functools import lru_cache
from multiprocessing.pool import ThreadPool
from typing import Tuple, List, Dict, Any, Optional, Iterator

import math
# noinspection PyUnresolvedReferences
from attr import dataclass
from copy import deepcopy
from enum import Enum
from pyspark import inheritable_thread_target, SparkContext
from pyspark import keyword_only
from pyspark.ml.common import inherit_doc
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.wrapper import JavaTransformer
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.pipelines.features.lgb_pipeline import SparkLGBSimpleFeatures
from sparklightautoml.reader.base import SparkToSparkReader
from sparklightautoml.tasks.base import SparkTask
from sparklightautoml.utils import log_exec_timer
from synapse.ml.lightgbm import LightGBMClassifier, LightGBMRegressor

from examples_utils import get_spark_session, get_dataset, prepare_test_and_train

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


def executors() -> List[str]:
    sc = SparkContext._active_spark_context
    return sc._jvm.org.apache.spark.lightautoml.utils.SomeFunctions.executors()


def cores_per_exec() -> int:
    return 6


@dataclass
class DatasetSlot:
    train_df: DataFrame
    test_df: DataFrame
    pref_locs: Optional[List[str]]
    num_tasks: int
    num_threads: int
    use_single_dataset_mode: bool
    free: bool


class ParallelismMode(Enum):
    pref_locs = 1
    no_single_dataset_mode = 2
    single_dataset_mode = 3
    no_parallelism = 4


class ParallelExperiment:
    def __init__(self,
                 spark: SparkSession,
                 dataset_name: str,
                 prepare_fold_num: int = 5,
                 use_fold_num: int = 5,
                 parallelism_mode: ParallelismMode = ParallelismMode.pref_locs):
        self.spark = spark
        self.dataset_name = dataset_name
        self.prepare_fold_num = prepare_fold_num
        self.use_fold_num = use_fold_num
        self.parallelism_mode = parallelism_mode
        self.base_dataset_path = f"/opt/spark_data/parallel_slama_{dataset_name}"
        self.train_path = os.path.join(self.base_dataset_path, "train.parquet")
        self.test_path = os.path.join(self.base_dataset_path, "test.parquet")
        self.metadata_path = os.path.join(self.base_dataset_path, "metadata.parquet")

        self._executors = list(executors())
        self._cores_per_exec = cores_per_exec()

        # for ParallelismMode.pref_locs
        self._train_slots: Optional[List[DatasetSlot]] = None
        self._train_slots_lock = threading.Lock()

        # for ParallelismMode.no_single_dataset_mode
        self._slot: Optional[DatasetSlot] = None

    def prepare_trains(self, max_job_parallelism: int):
        train_df = self.train_dataset
        test_df = self.test_dataset

        if self.parallelism_mode == ParallelismMode.pref_locs:
            execs_per_job = max(1, math.floor(len(self._executors) / max_job_parallelism))

            if len(self._executors) % max_job_parallelism != 0:
                warnings.warn(f"Uneven number of executors per job. Setting execs per job: {execs_per_job}.")

            slots_num = int(len(self._executors) / execs_per_job)

            self._train_slots = []

            def _coalesce_df_to_locs(df: DataFrame, pref_locs: List[str]):
                df = PrefferedLocsPartitionCoalescerTransformer(pref_locs=pref_locs).transform(df)
                df = df.cache()
                df.write.mode('overwrite').format('noop').save()
                return df

            for i in range(slots_num):
                pref_locs = self._executors[i * execs_per_job: (i + 1) * execs_per_job]

                # prepare train
                train_df = _coalesce_df_to_locs(train_df, pref_locs)

                # prepare test
                test_df = _coalesce_df_to_locs(test_df, pref_locs)

                self._train_slots.append(DatasetSlot(
                    train_df=train_df,
                    test_df=test_df,
                    pref_locs=pref_locs,
                    num_tasks=len(pref_locs) * self._cores_per_exec,
                    num_threads=-1,
                    use_single_dataset_mode=True,
                    free=True
                ))

                print(f"Pref lcos for slot #{i}: {pref_locs}")
        elif self.parallelism_mode == ParallelismMode.no_single_dataset_mode :
            num_tasks_per_job = max(1, math.floor(len(self._executors) * self._cores_per_exec / max_job_parallelism))
            self._slot = DatasetSlot(
                train_df=self.train_dataset,
                test_df=self.test_dataset,
                pref_locs=None,
                num_tasks=num_tasks_per_job,
                num_threads=-1,
                use_single_dataset_mode=False,
                free=False
            )
        elif self.parallelism_mode == ParallelismMode.single_dataset_mode:
            num_tasks_per_job = max(1, math.floor(len(self._executors) * self._cores_per_exec / max_job_parallelism))
            num_threads_per_exec = max(1, math.floor(num_tasks_per_job / len(self._executors)))

            if num_threads_per_exec != 1:
                warnings.warn(f"Num threads per exec {num_threads_per_exec} != 1. "
                              f"Overcommitting or undercommiting may happen due to "
                              f"uneven allocations of cores between executors for a job")

            self._slot = DatasetSlot(
                train_df=self.train_dataset,
                test_df=self.test_dataset,
                pref_locs=None,
                num_tasks=num_tasks_per_job,
                num_threads=num_threads_per_exec,
                use_single_dataset_mode=True,
                free=False
            )

        else:
            self._slot = DatasetSlot(
                train_df=self.train_dataset,
                test_df=self.test_dataset,
                pref_locs=None,
                num_tasks=self.train_dataset.rdd.getNumPartitions(),
                num_threads=-1,
                use_single_dataset_mode=True,
                free=False
            )

    @contextmanager
    def _use_train(self) -> Iterator[DatasetSlot]:
        if self.parallelism_mode == ParallelismMode.pref_locs:
            with self._train_slots_lock:
                free_slot = next((slot for slot in self._train_slots if slot.free))
                free_slot.free = False

            yield free_slot

            with self._train_slots_lock:
                free_slot.free = True
        else:
            yield self._slot

    def prepare_dataset(self, force=True):
        logger.info(f"Preparing dataset {self.dataset_name}. "
                    f"Writing train, test and metadata to {self.base_dataset_path}")
        return

        # if os.path.exists(self.base_dataset_path) and not force:
        #     logger.info(f"Found existing {self.base_dataset_path}. Skipping writing dataset files")
        #     return
        # elif os.path.exists(self.base_dataset_path):
        #     logger.info(f"Found existing {self.base_dataset_path}. "
        #                 f"Removing existing files because force is set to True")
        #     shutil.rmtree(self.base_dataset_path)

        seed = 42
        cv = self.prepare_fold_num
        dataset = get_dataset(self.dataset_name)

        train_df, test_df = prepare_test_and_train(dataset, seed)
        # train_df, test_df = handle_if_2stage(self.dataset_name, train_df), handle_if_2stage(self.dataset_name, test_df)

        task = SparkTask(task_type)

        sreader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False)
        spark_features_pipeline = SparkLGBSimpleFeatures()

        # prepare train
        train_sdataset = sreader.fit_read(train_df, roles=roles)
        train_sdataset = spark_features_pipeline.fit_transform(train_sdataset)

        # prepare test
        test_sdataset = sreader.read(test_df, add_array_attrs=True)
        test_sdataset = spark_features_pipeline.transform(test_sdataset)

        # os.makedirs(self.base_dataset_path)

        # train_sdataset.data.write.parquet("file://" + self.train_path)
        # test_sdataset.data.write.parquet("file://" + self.test_path)

        train_sdataset.data.write.parquet(self.train_path, mode='overwrite')
        test_sdataset.data.write.parquet(self.test_path, mode='overwrite')

        metadata = {
            "roles": train_sdataset.roles,
            "task_type": task_type,
            "target": roles["target"]
        }

        SparkSession.getActiveSession()\
            .createDataFrame([{"data": pickle.dumps(metadata)}]).write.parquet(self.metadata_path, mode='overwrite')

        logger.info(f"Dataset {self.dataset_name} has been prepared.")

    @property
    @lru_cache
    def train_dataset(self) -> DataFrame:
        exec_instances = int(self.spark.conf.get("spark.executor.instances"))
        cores_per_exec = int(self.spark.conf.get("spark.executor.cores"))
        # df = self.spark.read.parquet("file://" + self.train_path).repartition(exec_instances * cores_per_exec).cache()
        df = self.spark.read.parquet(self.train_path).repartition(exec_instances * cores_per_exec).cache()
        df.count()
        return df

    @property
    @lru_cache
    def test_dataset(self) -> DataFrame:
        exec_instances = int(self.spark.conf.get("spark.executor.instances"))
        cores_per_exec = int(self.spark.conf.get("spark.executor.cores"))
        # df = self.spark.read.parquet("file://" + self.test_path).repartition(exec_instances * cores_per_exec).cache()
        df = self.spark.read.parquet(self.test_path).repartition(exec_instances * cores_per_exec).cache()
        df.count()
        return df

    @property
    def metadata(self) -> Dict[str, Any]:
        data = self.spark.read.parquet(self.metadata_path).first().asDict()['data']
        return pickle.loads(data)

    def train_model(self, fold: int) -> Tuple[int, float]:
        logger.info(f"Starting to train the model for fold #{fold}")

        # train_df = self.train_dataset
        test_df = self.test_dataset
        md = self.metadata
        task_type = md["task_type"]

        test_df.sql_ctx.sparkSession.sparkContext.setLocalProperty("spark.scheduler.mode", "FAIR")
        # train_df.sql_ctx.sparkSession.sparkContext.setLocalProperty("spark.task.cpus", "6")

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
            inputCols=list(md['roles'].keys()),
            outputCol=f"LightGBM_vassembler_features",
            handleInvalid="keep"
        )

        with self._use_train() as slot:
            train_df, test_df, num_tasks, num_threads, use_single_dataset \
                = slot.train_df, slot.test_df, slot.num_tasks, slot.num_threads, slot.use_single_dataset_mode
            full_data = train_df.withColumn('is_val', sf.col('reader_fold_num') == fold)

            lgbm_booster = LightGBMRegressor if task_type == "reg" else LightGBMClassifier

            if num_threads != -1:
                params['numThreads'] = num_threads

            params["numIterations"] = 500

            lgbm = lgbm_booster(
                **params,
                featuresCol=assembler.getOutputCol(),
                labelCol=md['target'],
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

            transformer = lgbm.fit(assembler.transform(full_data))
            # metric_value = -1.0
            preds_df = transformer.transform(assembler.transform(test_df))

            score = SparkTask(task_type).get_dataset_metric()
            metric_value = score(
                preds_df.select(
                    SparkDataset.ID_COLUMN,
                    sf.col(md['target']).alias('target'),
                    sf.col(prediction_col).alias('prediction')
                )
            )

            logger.info(f"Finished training the model for fold #{fold}")

        return fold, metric_value

    def run(self, max_job_parallelism: int = 3) -> List[Tuple[int, float]]:
        with log_exec_timer("Parallel experiment runtime"):
            logger.info("Starting to run the experiment")

            tasks = [
                functools.partial(
                    self.train_model,
                    fold
                )
                for fold in range(self.use_fold_num)
            ]

            self.prepare_trains(max_job_parallelism)

            processes_num = 1 if self.parallelism_mode == ParallelismMode.no_parallelism else max_job_parallelism
            pool = ThreadPool(processes=processes_num)
            tasks = map(inheritable_thread_target, tasks)
            results = (result for result in pool.imap_unordered(lambda f: f(), tasks) if result)
            results = sorted(results, key=lambda x: x[0])

            logger.info("The experiment is finished")
            return results


def main():
    partitions_num = 6
    spark = get_spark_session(partitions_num=partitions_num)

    exp = ParallelExperiment(
        spark,
        dataset_name=os.environ.get("DATASET", "used_cars_dataset"),
        prepare_fold_num=int(os.environ.get("PREPARE_FOLD_NUM", "5")),
        use_fold_num=int(os.environ.get("USE_FOLD_NUM", "5")),
        parallelism_mode=ParallelismMode[os.environ.get("PARALLELISM_MODE", "pref_locs")]
    )
    exp.prepare_dataset()
    results = exp.run(max_job_parallelism=int(os.environ.get("MAX_JOB_PARALLELISM", "3")))

    for fold, metric_value in results:
        print(f"Metric value (fold = {fold}): {metric_value}")

    spark.stop()


if __name__ == "__main__":
    main()
