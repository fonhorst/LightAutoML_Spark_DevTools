import os
from typing import Tuple, Union

import mlflow
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from sparklightautoml.dataset.base import SparkDataset
from sparklightautoml.dataset.persistence import PlainCachePersistenceManager

# from sparklightautoml.validation.iterators import SparkFoldsIterator
# from sparklightautoml.validation.base import SparkBaseTrainValidIterator, TrainVal
# from sparklightautoml.pipelines.features.base import SparkFeaturesPipeline
# from sparklightautoml import mlwriters
# Important !!! CatBoost has problems with dependencies of SynapseML
# from synapse.ml.lightgbm import LightGBMClassificationModel
# from synapse.ml.lightgbm import LightGBMRegressionModel
# from synapse.ml.onnx import ONNXModel
from sparklightautoml.utils import log_exec_timer


def obtain_spark_session() -> SparkSession:
    if os.environ.get("SCRIPT_ENV", None) == "cluster":
        spark_sess = SparkSession.builder.getOrCreate()
    else:
        spark_sess = (
            SparkSession
                .builder
                .master("local[4]")
                .config("spark.jars.packages", "ai.catboost:catboost-spark_3.2_2.12:1.2")
                .config("spark.jars", "../../../SLAMA/jars/spark-lightautoml_2.12-0.1.1.jar")
                .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")
                .config("spark.executor.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .config("spark.kryoserializer.buffer.max", "512m")
                .config("spark.cleaner.referenceTracking.cleanCheckpoints", "true")
                .config("spark.cleaner.referenceTracking", "true")
                .config("spark.cleaner.periodicGC.interval", "1min")
                .config("spark.sql.shuffle.partitions", f"{4}")
                .config("spark.driver.memory", "4g")
                .config("spark.executor.memory", "4g")
                .config("spark.task.cpus", "4")
                .config("spark.sql.execution.arrow.pyspark.enabled", "true")
                .config("spark.sql.autoBroadcastJoinThreshold", "-1")
                .getOrCreate()
        )
    return spark_sess


def train_test_split(dataset: SparkDataset,
                     exec_instances: int,
                     exec_cores: int,
                     test_slice_or_fold_num: Union[float, int] = 0.2) \
        -> Tuple[SparkDataset, SparkDataset]:

    if isinstance(test_slice_or_fold_num, float):
        assert 0 <= test_slice_or_fold_num <= 1
        train, test = dataset.data.randomSplit([1 - test_slice_or_fold_num, test_slice_or_fold_num])
    else:
        train = dataset.data.where(sf.col(dataset.folds_column) != test_slice_or_fold_num).repartition(
            exec_cores * exec_instances
        )
        test = dataset.data.where(sf.col(dataset.folds_column) == test_slice_or_fold_num).repartition(
            exec_cores * exec_instances
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


if __name__ == "__main__":
    spark = obtain_spark_session()
    import catboost_spark

    exec_instances = 4 if spark.sparkContext.master.startswith("local") \
        else int(spark.conf.get("spark.executor.instances"))

    exec_cores = 4 if spark.sparkContext.master.startswith("local") \
        else int(spark.conf.get("spark.executor.cores"))

    dataset_name = os.environ.get("DATASET", "used_cars_dataset")
    parallelism = int(os.environ.get("EXP_JOB_PARALLELISM", "1"))
    feat_pipe = "lgb_adv"
    dataset_path = f"file:///opt/spark_data/preproccessed_datasets/{dataset_name}__{feat_pipe}__features.dataset"
    default_params = {
        # "task_type": "CPU",
        "threadCount": exec_cores,
        "randomSeed": 42,
        "iterations": 500, # "num_trees": 3000,
        "earlyStoppingRounds": 50_000,
        "learningRate": 0.05,
        "l2LeafReg": 1e-2,
        "bootstrapType": catboost_spark.EBootstrapType.Bernoulli,
        # "grow_policy": "SymmetricTree",
        "depth":  5, # "max_depth": 5,
        # "min_data_in_leaf": 1,
        "oneHotMaxSize": 10,
        "foldPermutationBlock": 1,
        # "boosting_type": "Plain",
        # "boost_from_average": True,
        "odType": catboost_spark.EOverfittingDetectorType.Iter,
        # "odWait": 100,
        # "max_bin": 32,
        "featureBorderType": catboost_spark.EBorderSelectionType.GreedyLogSum,
        "nanMode": catboost_spark.ENanMode.Min,
        # "verbose": 100,
        "allowWritingFiles": False
    }

    # load and prepare data
    ds = SparkDataset.load(
        path=dataset_path,
        persistence_manager=PlainCachePersistenceManager()
    )

    train_ds, test_ds = train_test_split(ds, exec_instances=exec_instances,
                                         exec_cores=exec_cores, test_slice_or_fold_num=4)
    assembler = VectorAssembler(inputCols=ds.features, outputCol="features", handleInvalid="keep")

    with mlflow.start_run(experiment_id=os.environ["EXPERIMENT"]):
        mlflow.log_params({
            "app_id": spark.sparkContext.applicationId,
            "app_name": spark.sparkContext.appName,
            "dataset": dataset_name,
            "ml_algo_name": "cb",
            "dataset_path": dataset_path,
            "feat_pipe": feat_pipe,
            "mlalgo_default_params": default_params,
            "exec_instances": spark.conf.get('spark.executor.instances', "-1"),
            "exec_cores": spark.conf.get('spark.executor.cores', "-1"),
        })
        mlflow.log_dict(dict(spark.sparkContext.getConf().getAll()), "spark_conf.json")

        with log_exec_timer("ml_algo_time") as fit_timer:
            trainPool = catboost_spark.Pool(
                assembler.transform(train_ds.data).select("features", sf.col(ds.target_column).alias('label'))
            )
            evalPool = catboost_spark.Pool(
                assembler.transform(test_ds.data).select("features", sf.col(ds.target_column).alias('label'))
            )

            if ds.task.name == "binary":
                classifier = catboost_spark.CatBoostClassifier(**default_params)
            else:
                classifier = catboost_spark.CatBoostRegressor(**default_params)

            # train a model
            model = classifier.fit(trainPool, evalDatasets=[evalPool])

        mlflow.log_metric(fit_timer.name, fit_timer.duration)

        with log_exec_timer("oof_score_time") as oof_timer:
            # apply the model
            predictions = model.transform(evalPool.data)
            predictions.write.mode("overwrite").format("noop").save()

        mlflow.log_metric(oof_timer.name, oof_timer.duration)
