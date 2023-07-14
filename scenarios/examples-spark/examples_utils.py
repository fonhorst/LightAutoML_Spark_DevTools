import inspect
import os
import itertools
from typing import Tuple, Optional, Union, List, Callable, Dict, Any

import mlflow
from dataclasses import dataclass, field
from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from sparklightautoml.computations.base import ComputationSlot
from sparklightautoml.computations.parallel import ParallelComputationsSession, ParallelComputationsManager
from sparklightautoml.computations.utils import get_executors, get_executors_cores
from sparklightautoml.dataset import persistence
from sparklightautoml.dataset.base import PersistenceManager, PersistableDataFrame, SparkDataset, PersistenceLevel
from sparklightautoml.ml_algo.boost_lgbm import SparkBoostLGBM
from sparklightautoml.ml_algo.linear_pyspark import SparkLinearLBFGS
from sparklightautoml.utils import SparkDataFrame, log_exec_timer

BUCKET_NUMS = 16
PERSISTENCE_MANAGER_ENV_VAR = "PERSISTENCE_MANAGER"
BASE_DATASETS_PATH = "file:///opt/spark_data/"


@dataclass(frozen=True)
class Dataset:
    path: str
    task_type: str
    roles: Dict[str, Any]
    dtype: Dict[str, str] = field(default_factory=dict)
    file_format: str = 'csv'
    file_format_options: Dict[str, Any] = field(default_factory=lambda: {"header": True, "escape": "\""})

    def load(self) -> SparkDataFrame:
        spark = SparkSession.getActiveSession()
        return spark.read.format(self.file_format).options(**self.file_format_options).load(self.path)


def ds_path(rel_path: str) -> str:
    return os.path.join(BASE_DATASETS_PATH, rel_path)


def rows_to_name(rows_count: int) -> str:
    num = int(rows_count / 1_000_000)
    if (rows_count % 1_000_000 == 0) and num >= 1:
        return f"{num}m"

    num = int(rows_count / 1_000)
    if (rows_count % 1_000 == 0) and num >= 1:
        return f"{num}k"

    raise ValueError(f"Invalid count to make it into an abbreviation: {rows_count}")


used_cars_params = {
    "task_type": "reg",
    "roles": {
        "target": "price",
        "drop": ["dealer_zip", "description", "listed_date",
                 "year", 'Unnamed: 0', '_c0',
                 'sp_id', 'sp_name', 'trimId',
                 'trim_name', 'major_options', 'main_picture_url',
                 'interior_color', 'exterior_color'],
        "numeric": ['latitude', 'longitude', 'mileage']
    },
    "dtype": {
        'fleet': 'str', 'frame_damaged': 'str',
        'has_accidents': 'str', 'isCab': 'str',
        'is_cpo': 'str', 'is_new': 'str',
        'is_oemcpo': 'str', 'salvage': 'str', 'theft_title': 'str', 'franchise_dealer': 'str'
    }
}

TEST_CARDINALITY_DATASETS = {
    f"test_cardinality_{rows_to_name(rows)}_{cols}c": Dataset(
        path=ds_path(f'data_for_LE_TE_tests/{rows}_rows_{cols}_columns_id_with_reg_target.parquet'),
        roles={'target': 'target'},
        task_type='reg',
        file_format="parquet",
        file_format_options={}
    ) for rows, cols in itertools.product([100000, 250000, 500000, 1000000], [10, 100, 200, 500])
}


DATASETS = {
    "used_cars_dataset": Dataset(
        path=ds_path("small_used_cars_data.csv"),
        **used_cars_params
    ),
    "used_cars_dataset_07x": Dataset(
        path=ds_path("derivative_datasets/07x_dataset.csv"),
        **used_cars_params
    ),
    "used_cars_dataset_1x": Dataset(
        path=ds_path("derivative_datasets/1x_dataset.csv"),
        **used_cars_params
    ),
    "used_cars_dataset_4x": Dataset(
        path=ds_path("derivative_datasets/4x_dataset.csv"),
        **used_cars_params
    ),
    "used_cars_dataset_10x": Dataset(
        path=ds_path("derivative_datasets/10x_dataset.csv"),
        **used_cars_params
    ),
    "lama_test_dataset": Dataset(
        path=ds_path("sampled_app_train.csv"),
        task_type="binary",
        roles={"target": "TARGET", "drop": ["SK_ID_CURR"]}
    ),
    # https://www.openml.org/d/4549
    "buzz_dataset": Dataset(
        path=ds_path("Buzzinsocialmedia_Twitter_25k.csv"),
        task_type="binary",
        roles={"target": "TARGET", "drop": ["SK_ID_CURR"]}
    ),
    # https://www.openml.org/d/734
    "ailerons_dataset": Dataset(
        path=ds_path("ailerons.csv"),
        task_type="binary",
        roles={"target": "binaryClass"}
    ),
    # https://www.openml.org/d/382
    "ipums_97": Dataset(
        path=ds_path("ipums_97.csv"),
        task_type="multiclass",
        roles={"target": "movedin"}
    ),

    "company_bankruptcy_dataset": Dataset(
        path=ds_path("company_bankruptcy_prediction_data.csv"),
        task_type="binary",
        roles={"target": "Bankrupt?"}
    ),

    "msd_2stage": Dataset(
        path="hdfs://node21.bdcl:9000/opt/spark_data/replay/experiments/msd_first_level_80_20/combined_train_4models.parquet",
        task_type="binary",
        roles={"target": "target", "drop": ["user_idx", "item_idx"]},
        file_format="parquet",
        file_format_options={}
    ),

    "ml25m_2stage": Dataset(
        path="hdfs://node21.bdcl:9000/opt/spark_data/replay/experiments/ml25m_first_level_80_20/combined_train_4models.parquet",
        task_type="binary",
        # SparkLightAutoML-0.3.0-py3-none-any.whl/sparklightautoml/reader/base.py", line 565, in _guess_role
        # KeyError: 'timestamp
        roles={"target": "target", "drop": ["user_idx", "item_idx",
                                            "i_max_interact_date", "u_min_interact_date",
                                            "u_max_interact_date", "i_min_interact_date"]},
        file_format="parquet",
        file_format_options={}
    ),

    "ml25m_0035p_2stage": Dataset(
        path="hdfs://node21.bdcl:9000/opt/spark_data/replay/experiments/ml25m_first_level_80_20/combined_train_4models_035percent.parquet",
        task_type="binary",
        # SparkLightAutoML-0.3.0-py3-none-any.whl/sparklightautoml/reader/base.py", line 565, in _guess_role
        # KeyError: 'timestamp
        roles={"target": "target", "drop": ["user_idx", "item_idx",
                                            "i_max_interact_date", "u_min_interact_date",
                                            "u_max_interact_date", "i_min_interact_date"]},
        file_format="parquet",
        file_format_options={}
    ),

    "ml25m_010p_2stage": Dataset(
        path="hdfs://node21.bdcl:9000/opt/spark_data/replay/experiments/ml25m_first_level_80_20/combined_train_4models_10percent.parquet",
        task_type="binary",
        # SparkLightAutoML-0.3.0-py3-none-any.whl/sparklightautoml/reader/base.py", line 565, in _guess_role
        # KeyError: 'timestamp
        roles={"target": "target", "drop": ["user_idx", "item_idx",
                                            "i_max_interact_date", "u_min_interact_date",
                                            "u_max_interact_date", "i_min_interact_date"]},
        file_format="parquet",
        file_format_options={}
    ),

    "ml25m_035p_2stage": Dataset(
        path="hdfs://node21.bdcl:9000/opt/spark_data/replay/experiments/ml25m_first_level_80_20/combined_train_4models_35percent.parquet",
        task_type="binary",
        # SparkLightAutoML-0.3.0-py3-none-any.whl/sparklightautoml/reader/base.py", line 565, in _guess_role
        # KeyError: 'timestamp
        roles={"target": "target", "drop": ["user_idx", "item_idx",
                                               "i_max_interact_date", "u_min_interact_date",
                                               "u_max_interact_date", "i_min_interact_date"]},
        file_format="parquet",
        file_format_options={}
    ),

    "higgs":Dataset(
        path=ds_path("higgs_training.csv"),
        task_type="binary",
        roles={'target': 'Label', 'drop': ['EventId']}
    ),

    "kaggle_a":Dataset(
        path=ds_path("kaggle-a/train_data"),
        task_type="binary",
        roles={'target': 'team_A_scoring_within_10sec', 'drop': ['team_B_scoring_within_10sec']}
    ),

    "kaggle_a_2x":Dataset(
        path=ds_path("kaggle-a/train_data_2x"),
        task_type="binary",
        roles={'target': 'team_A_scoring_within_10sec', 'drop': ['team_B_scoring_within_10sec']}
    ),

    "kaggle_a_1.5x":Dataset(
        path=ds_path("kaggle-a/train_data_1.5x"),
        task_type="binary",
        roles={'target': 'team_A_scoring_within_10sec', 'drop': ['team_B_scoring_within_10sec']}
    ),

    "synth_10kk_100": Dataset(
        path=ds_path('synth_datasets/synth_dataset_classification_n_10_000_000_f_100.csv'),
        roles={'target': 'target'},
        task_type='binary'
    ),

    "synth_7kk_100": Dataset(
        path=ds_path('synth_datasets/synth_dataset_classification_n_7_000_000_f_100_part_of_10mln.csv'),
        roles={'target': 'target'},
        task_type='binary'
    ),

    "synth_5kk_100": Dataset(
        path=ds_path('synth_datasets/synth_dataset_classification_n_5_000_000_f_100_part_of_10mln.csv'),
        roles={'target': 'target'},
        task_type='binary'
    ),

    "synth_1kk_100": Dataset(
        path=ds_path('synth_datasets/synth_dataset_classification_n_1_000_000_f_100.csv'),
        roles={'target': 'target'},
        task_type='binary'
    ),

    **TEST_CARDINALITY_DATASETS
}


def get_dataset(name: str) -> Dataset:
    assert name in DATASETS, f"Unknown dataset: {name}. Known datasets: {list(DATASETS.keys())}"
    return DATASETS[name]


def prepare_test_and_train(
        dataset: Dataset,
        seed: int,
        test_size: float = 0.2
) -> Tuple[SparkDataFrame, SparkDataFrame]:
    assert 0 <= test_size <= 1

    spark = SparkSession.getActiveSession()

    execs = int(spark.conf.get('spark.executor.instances', '1'))
    cores = int(spark.conf.get('spark.executor.cores', '8'))

    data = dataset.load()

    data = data.repartition(execs * cores).cache()
    data.write.mode('overwrite').format('noop').save()

    train_data, test_data = data.randomSplit([1 - test_size, test_size], seed)
    train_data = train_data.cache()
    test_data = test_data.cache()
    train_data.write.mode('overwrite').format('noop').save()
    test_data.write.mode('overwrite').format('noop').save()

    data.unpersist()

    return train_data, test_data


def get_spark_session(partitions_num: Optional[int] = None):
    partitions_num = partitions_num if partitions_num else BUCKET_NUMS

    if os.environ.get("SCRIPT_ENV", None) == "cluster":
        # spark_sess = SparkSession.builder.config("spark.locality.wait", "30s").getOrCreate()
        spark_sess = SparkSession.builder.getOrCreate()
    else:
        # TODO: SLAMA - fix .config("spark.jars", "../../LightAutoML/jars/spark-lightautoml_2.12-0.1.jar") with correct path
        spark_sess = (
            SparkSession
                .builder
                .master("local[4]")
                # .config("spark.jars.packages",
                #         "com.microsoft.azure:synapseml_2.12:0.9.5,io.github.fonhorst:spark-lightautoml_2.12:0.1")
                .config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.5")
                .config("spark.jars", "../../../SLAMA/jars/spark-lightautoml_2.12-0.1.1.jar")
                .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
                .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")
                .config("spark.executor.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .config("spark.kryoserializer.buffer.max", "512m")
                .config("spark.cleaner.referenceTracking.cleanCheckpoints", "true")
                .config("spark.cleaner.referenceTracking", "true")
                .config("spark.cleaner.periodicGC.interval", "1min")
                .config("spark.sql.shuffle.partitions", f"{partitions_num}")
                # .config("spark.default.parallelism", f"{partitions_num}")
                .config("spark.driver.memory", "4g")
                .config("spark.executor.memory", "4g")
                .config("spark.sql.execution.arrow.pyspark.enabled", "true")
                .config("spark.sql.autoBroadcastJoinThreshold", "-1")
                .getOrCreate()
        )

    spark_sess.sparkContext.setCheckpointDir("/tmp/spark_checkpoints")

    spark_sess.sparkContext.setLogLevel("WARN")

    return spark_sess


def get_persistence_manager(run_id: str, name: Optional[str] = None):
    spark = SparkSession.getActiveSession()

    computed_bucket_nums = BUCKET_NUMS if spark.sparkContext.master.startswith('local') \
        else int(spark.conf.get('spark.executor.instances')) * int(spark.conf.get('spark.executor.cores'))
    bucket_nums = int(os.environ.get("BUCKET_NUMS", computed_bucket_nums))

    bucketed_folder = f"/tmp/spark_buckets/run_{run_id}"

    # script_env = os.environ.get("SCRIPT_ENV", "local")
    # if script_env == "cluster":
    #     hdfs_url = os.environ.get("SCRIPT_HDFS_URL", "http://node21.bdcl:9870")
    #     client = hdfs.InsecureClient(hdfs_url)
    #     client.makedirs(bucketed_folder)
    # else:
    #     os.makedirs(bucketed_folder)

    arg_vals = {
        "bucketed_datasets_folder": bucketed_folder,
        "bucket_nums": bucket_nums
    }

    class_name = name or os.environ.get(PERSISTENCE_MANAGER_ENV_VAR, None) or "CompositeBucketedPersistenceManager"
    clazz = getattr(persistence, class_name)
    sig = inspect.signature(getattr(clazz, "__init__"))

    ctr_arg_vals = {
        name: arg_vals.get(name, None if p.default is p.empty else p.default)
        for name, p in sig.parameters.items() if name != 'self'
    }

    none_val_args = [name for name, val in ctr_arg_vals.items() if val is None]
    assert len(none_val_args) == 0, f"Cannot instantiate class {class_name}. " \
                                    f"Values for the following arguments have not been found: {none_val_args}"

    pmanager = MLflowWrapperPersistenceManager(clazz(**ctr_arg_vals))

    mlflow.log_param("persistence_manager", clazz.__name__)

    return pmanager


def check_executors_count():
    """
    Checks the number of executors matches the param 'spark.executor.instances' if this param is set
    https://spark.apache.org/docs/latest/monitoring.html#rest-api
    """
    pass
    # spark = SparkSession.getActiveSession()
    # exec_instances = int(spark.conf.get("spark.executor.instances", None))
    # if exec_instances:
    #     # doing it to verify that computations is possible
    #     spark.sparkContext.parallelize(list(range(10))).sum()
    #
    #     print(f"WebUiUrl: {spark.sparkContext.uiWebUrl}")
    #
    #     port = int(spark.sparkContext.uiWebUrl.split(':')[2])
    #
    #     url = f"http://localhost:{port}/api/v1/applications/{spark.sparkContext.applicationId}/allexecutors"
    #     with urllib.request.urlopen(url) as url:
    #         response = url.read().decode()
    #         print("=============")
    #         print(response)
    #         print("=============")
    #         data = json.loads(response)
    #
    #     assert len(data) - 1 == exec_instances, \
    #         f"Incorrect number of executors. Expected: {exec_instances}. Found: {len(data) - 1}"


def log_session_params_to_mlflow():
    spark = SparkSession.getActiveSession()

    mlflow.log_params({
        "app_id": spark.sparkContext.applicationId,
        "app_name": spark.sparkContext.appName,
        "executors": spark.sparkContext.getConf().get("spark.executor.instances", "-1"),
        "executor_cores": spark.sparkContext.getConf().get("spark.executor.cores", "-1"),
        "executor_memory": spark.sparkContext.getConf().get("spark.executor.memory", "-1"),
        "partitions_nums": spark.conf.get("spark.default.parallelism", None),
        "bucket_nums": os.environ.get("BUCKET_NUMS", None)
    })

    mlflow.log_dict(dict(spark.sparkContext.getConf().getAll()), "spark_conf.json")


class MLflowWrapperPersistenceManager(PersistenceManager):
    def __init__(self, instance: PersistenceManager):
        self._instance = instance

    @property
    def uid(self) -> str:
        return self._instance.uid

    @property
    def children(self) -> List['PersistenceManager']:
        return self._instance.children

    @property
    def datasets(self) -> List[SparkDataset]:
        return self._instance.datasets

    @property
    def all_datasets(self) -> List[SparkDataset]:
        return self._instance.all_datasets

    def persist(self, dataset: Union[SparkDataset, PersistableDataFrame],
                level: PersistenceLevel = PersistenceLevel.REGULAR) -> PersistableDataFrame:

        if level == PersistenceLevel.READER:
            with log_exec_timer("reader_time") as timer:
                result = self._instance.persist(dataset, level)

            mlflow.log_metric(timer.name, timer.duration)
        else:
            result = self._instance.persist(dataset, level)

        return result

    def unpersist(self, uid: str):
        self._instance.unpersist(uid)

    def unpersist_all(self):
        self._instance.unpersist_all()

    def unpersist_children(self):
        self._instance.unpersist_children()

    def child(self) -> 'PersistenceManager':
        return self._instance.child()

    def remove_child(self, child: Union['PersistenceManager', str]):
        self._instance.remove_child(child)

    def is_persisted(self, pdf: PersistableDataFrame) -> bool:
        return self._instance.is_persisted(pdf)


class mlflow_log_exec_timer(log_exec_timer):
    def __init__(self, name: Optional[str] = None):
        super(mlflow_log_exec_timer, self).__init__(name)
        
    def __exit__(self, typ, value, traceback):
        super().__exit__(typ, value, traceback)
        if self.name:
            mlflow.log_metric(self.name, self.duration)


def initialize_environment(main: Callable[[SparkSession], None]):
    def func():
        spark = get_spark_session()
        check_executors_count()

        # log_to_mlflow = bool(int(os.environ.get("LOG_TO_MLFLOW", "0")))
        #
        # if log_to_mlflow:
        #     exp_id = os.environ.get("EXPERIMENT", None)
        #     assert exp_id, "EXPERIMENT should be set if LOG_TO_MLFLOW is true"
        #     with mlflow.start_run(experiment_id=exp_id):
        #         log_session_params_to_mlflow()
        #         main(spark)
        # else:
        main(spark)

    return func


def handle_if_2stage(dataset_name: str, df: SparkDataFrame) -> SparkDataFrame:
    if dataset_name.endswith("2stage"):
        def explode_vec(col_name: str):
            return [sf.col(col_name).getItem(i).alias(f'{col_name}_{i}') for i in range(100)]

        df = df.select(
            "*", *explode_vec("user_factors"), #*explode_vec("item_factors"),
            #*explode_vec("factors_mult")
        ).drop("user_factors", "item_factors", "factors_mult")

    return df


def train_test_split(dataset: SparkDataset, test_slice_or_fold_num: Union[float, int] = 0.2) \
        -> Tuple[SparkDataset, SparkDataset]:

    if isinstance(test_slice_or_fold_num, float):
        assert 0 <= test_slice_or_fold_num <= 1
        train, test = dataset.data.randomSplit([1 - test_slice_or_fold_num, test_slice_or_fold_num])
    else:
        train = dataset.data.where(sf.col(dataset.folds_column) != test_slice_or_fold_num).repartition(
            len(get_executors()) * get_executors_cores()
        )
        test = dataset.data.where(sf.col(dataset.folds_column) == test_slice_or_fold_num).repartition(
            len(get_executors()) * get_executors_cores()
        )

    train, test = train.cache(), test.cache()
    train.write.mode("overwrite").format("noop").save()
    test.write.mode("overwrite").format("noop").save()

    rows = (
        train
        .withColumn("__partition_id__", sf.spark_partition_id())
        .groupby("__partition_id__").agg(sf.count("*").alias("all_values"))
        .collect()
    )
    for row in rows:
        assert row["all_values"] != 0, f"Empty partitions: {row['__partition_id_']}"

    train_dataset, test_dataset = dataset.empty(), dataset.empty()
    train_dataset.set_data(train, dataset.features, roles=dataset.roles)
    test_dataset.set_data(test, dataset.features, roles=dataset.roles)

    return train_dataset, test_dataset


class ReportingParallelComputeSession(ParallelComputationsSession):
    def __init__(self, dataset: SparkDataset, parallelism: int, use_location_prefs_mode: int):
        super().__init__(dataset, parallelism, use_location_prefs_mode)
        self.prepare_dataset_time: Optional[float] = None

    def _make_slots_on_dataset_copies_coalesced_into_preffered_locations(self, dataset: SparkDataset) \
            -> List[ComputationSlot]:
        with log_exec_timer("prepare_dataset_with_locations_prefferences") as timer:
            result = super()._make_slots_on_dataset_copies_coalesced_into_preffered_locations(dataset)
        self.prepare_dataset_time = timer.duration
        return result


class ReportingParallelComputionsManager(ParallelComputationsManager):
    def __init__(self, parallelism: int = 1, use_location_prefs_mode: bool = False):
        super().__init__(parallelism, use_location_prefs_mode)
        self.last_session: Optional[ReportingParallelComputeSession] = None

    def session(self, dataset: Optional[SparkDataset] = None) -> ParallelComputationsSession:
        self.last_session = ReportingParallelComputeSession(dataset, self._parallelism, self._use_location_prefs_mode)
        return self.last_session


def get_ml_algo():
    ml_algo_name = os.environ.get("EXP_ML_ALGO", "lgb")

    if ml_algo_name == "linear_l2":
        feat_pipe = "linear"  # linear, lgb_simple or lgb_adv
        default_params = {'regParam': [1e-5], "maxIter": 100, "aggregationDepth": 2, "tol": 0.0}
        ml_algo = SparkLinearLBFGS(default_params)
        tag = ml_algo_name
    elif ml_algo_name == "lgb":
        use_single_dataset_mode = int(os.environ.get("EXP_LGB_SINGLE_DATASET_MODE", "1")) == 1
        feat_pipe = "lgb_adv"  # linear, lgb_simple or lgb_adv
        default_params = {"numIterations": 500, "earlyStoppingRound": 50_000}
        ml_algo = SparkBoostLGBM(default_params,
                                 use_barrier_execution_mode=True,
                                 use_single_dataset_mode=use_single_dataset_mode)
        tag = ml_algo_name if use_single_dataset_mode else f"{ml_algo_name}_no_single_dataset_mode"
    else:
        raise ValueError(f"Unknown ml algo: {ml_algo_name}")

    return feat_pipe, default_params, ml_algo, ml_algo_name, tag
# def get_ml_algo():
#     ml_algo_name = os.environ.get("EXP_ML_ALGO", "linear_l2")
#
#     if ml_algo_name == "linear_l2":
#         feat_pipe = "linear"  # linear, lgb_simple or lgb_adv
#         default_params = {'regParam': [1e-5], "maxIter": 100, "aggregationDepth": 2, "tol": 0.0}
#         ml_algo = SparkLinearLBFGS(default_params)
#     elif ml_algo_name == "lgb":
#         feat_pipe = "lgb_adv"  # linear, lgb_simple or lgb_adv
#         default_params = {"numIterations": 500, "earlyStoppingRound": 50_000}
#         ml_algo = SparkBoostLGBM(default_params, use_barrier_execution_mode=True)
#     else:
#         raise ValueError(f"Unknown ml algo: {ml_algo_name}")
#
#     return feat_pipe, default_params, ml_algo


def check_allocated_executors():
    spark = SparkSession.getActiveSession()
    if not spark.sparkContext.master.startswith("local"):
        allocated_executos_num = len(get_executors())
        assert allocated_executos_num == int(spark.conf.get("spark.executor.instances")), \
            f"Not all desired executors have been allocated: " \
            f"{allocated_executos_num} != {spark.conf.get('spark.executor.instances')}"


def check_columns(original_df: SparkDataFrame, predicts_df: SparkDataFrame):
    absent_columns = set(original_df.columns).difference(predicts_df.columns)
    assert len(absent_columns) == 0, \
        f"Some columns of the original dataframe is absent from the processed dataset: {absent_columns}"
