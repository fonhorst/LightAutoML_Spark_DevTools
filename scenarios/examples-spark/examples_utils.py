import inspect
import os
from typing import Tuple, Optional, Union, List, Callable

import mlflow
from pyspark.sql import SparkSession
from sparklightautoml.dataset import persistence
from sparklightautoml.dataset.base import PersistenceManager, PersistableDataFrame, SparkDataset, PersistenceLevel
from sparklightautoml.utils import SparkDataFrame, log_exec_timer

BUCKET_NUMS = 16
PERSISTENCE_MANAGER_ENV_VAR = "PERSISTENCE_MANAGER"


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

DATASETS = {
    "used_cars_dataset": {
            "path": "file:///opt/spark_data/small_used_cars_data.csv",
            **used_cars_params
    },

    "used_cars_dataset_1x": {
        "path": "file:///opt/spark_data/derivative_datasets/1x_dataset.csv",
        **used_cars_params
    },

    "used_cars_dataset_4x": {
        "path": "file:///opt/spark_data/derivative_datasets/4x_dataset.csv",
        **used_cars_params
    },

    # https://www.openml.org/d/4549
    "buzz_dataset": {
        "path": "file:///opt/spark_data/Buzzinsocialmedia_Twitter_25k.csv",
        "task_type": "reg",
        "roles": {"target": "Annotation"},
    },

    "lama_test_dataset": {
        "path": "file:///opt/spark_data/sampled_app_train.csv",
        "task_type": "binary",
        "roles": {"target": "TARGET", "drop": ["SK_ID_CURR"]},
    },

    # https://www.openml.org/d/734
    "ailerons_dataset": {
        "path": "file:///opt/spark_data/ailerons.csv",
        "task_type": "binary",
        "roles": {"target": "binaryClass"},
    },

    # https://www.openml.org/d/382
    "ipums_97": {
        "path": "file:///opt/spark_data/ipums_97.csv",
        "task_type": "multiclass",
        "roles": {"target": "movedin"},
    },

    "company_bankruptcy_dataset": {
        "path": "file:///opt/spark_data/company_bankruptcy_prediction_data.csv",
        "task_type": "binary",
        "roles": {"target": "Bankrupt?"},
    },

    "msd_2stage": {
        "path": "file:///opt/spark_data/sampled_app_train.csv",
        "task_type": "binary",
        "roles": {"target": "TARGET", "drop": ["SK_ID_CURR"]},
    },
}


def get_dataset_attrs(name: str):
    return (
        DATASETS[name]['path'],
        DATASETS[name]['task_type'],
        DATASETS[name]['roles'],
        # to assure that LAMA correctly interprets certain columns as categorical
        DATASETS[name].get('dtype', dict()),
    )


def prepare_test_and_train(spark: SparkSession, path: str, seed: int) -> Tuple[SparkDataFrame, SparkDataFrame]:
    execs = int(spark.conf.get('spark.executor.instances', '1'))
    cores = int(spark.conf.get('spark.executor.cores', '8'))

    data = spark.read.csv(path, header=True, escape="\"")
    data = data.repartition(execs * cores).cache()
    data.write.mode('overwrite').format('noop').save()

    train_data, test_data = data.randomSplit([0.8, 0.2], seed)
    train_data = train_data.cache()
    test_data = test_data.cache()
    train_data.write.mode('overwrite').format('noop').save()
    test_data.write.mode('overwrite').format('noop').save()

    data.unpersist()

    return train_data, test_data


def get_spark_session(partitions_num: Optional[int] = None):
    partitions_num = partitions_num if partitions_num else BUCKET_NUMS

    if os.environ.get("SCRIPT_ENV", None) == "cluster":
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
                .config("spark.jars", "../../../LightAutoML/jars/spark-lightautoml_2.12-0.1.jar")
                .config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven")
                .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")
                .config("spark.executor.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .config("spark.kryoserializer.buffer.max", "512m")
                .config("spark.cleaner.referenceTracking.cleanCheckpoints", "true")
                .config("spark.cleaner.referenceTracking", "true")
                .config("spark.cleaner.periodicGC.interval", "1min")
                .config("spark.sql.shuffle.partitions", f"{partitions_num}")
                .config("spark.default.parallelism", f"{partitions_num}")
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

    mlflow.log_param("application_id", spark.sparkContext.applicationId)
    mlflow.log_param("executors", spark.conf.get("spark.executor.instances", None))
    mlflow.log_param("executor_cores", spark.conf.get("spark.executor.cores", None))
    mlflow.log_param("executor_memory", spark.conf.get("spark.executor.memory", None))
    mlflow.log_param("partitions_nums", spark.conf.get("spark.default.parallelism", None))
    mlflow.log_param("bucket_nums", os.environ.get("BUCKET_NUMS", None))
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


def mlflow_deco(main: Callable[[int, int, str], None]):
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
