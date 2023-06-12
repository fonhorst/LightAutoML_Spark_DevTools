import os
from typing import Tuple, Union

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


def train_test_split(dataset: SparkDataset, test_slice_or_fold_num: Union[float, int] = 0.2) \
        -> Tuple[SparkDataset, SparkDataset]:

    spark = SparkSession.getActiveSession()
    exec_instances = int(spark.conf.get('spark.executor.instances', '1'))

    if isinstance(test_slice_or_fold_num, float):
        assert 0 <= test_slice_or_fold_num <= 1
        train, test = dataset.data.randomSplit([1 - test_slice_or_fold_num, test_slice_or_fold_num])
    else:
        train = dataset.data.where(sf.col(dataset.folds_column) != test_slice_or_fold_num).repartition(
            1 * exec_instances
        )
        test = dataset.data.where(sf.col(dataset.folds_column) == test_slice_or_fold_num).repartition(
            1 * exec_instances
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
    import catboost_spark

    dataset_name = os.environ.get("DATASET", "lama_test_dataset")
    parallelism = int(os.environ.get("EXP_JOB_PARALLELISM", "1"))
    feat_pipe = "lgb_adv"
    n_trials = 64
    timeout = 60000
    stabilize = False
    dataset_path = f"file:///opt/spark_data/preproccessed_datasets/{dataset_name}__{feat_pipe}__features.dataset"

    # load and prepare data
    ds = SparkDataset.load(
        path=dataset_path,
        persistence_manager=PlainCachePersistenceManager()
    )

    train_ds, test_ds = train_test_split(ds, test_slice_or_fold_num=4)
    assembler = VectorAssembler(inputCols=ds.features, outputCol="features", handleInvalid="keep")

    trainPool = catboost_spark.Pool(
        assembler.transform(train_ds.data).select("features", sf.col(ds.target_column).alias('label'))
    )
    evalPool = catboost_spark.Pool(
        assembler.transform(test_ds.data).select("features", sf.col(ds.target_column).alias('label'))
    )
    classifier = catboost_spark.CatBoostClassifier()

    # train a model
    model = classifier.fit(trainPool, evalDatasets=[evalPool])

    # apply the model
    predictions = model.transform(evalPool.data)
    predictions.show()
