import os

from lightfm import LightFM
from pyspark.sql import functions as sf, SparkSession, Window
from pyspark.sql.types import IntegerType
from pyspark.conf import SparkConf
from replay.session_handler import State, get_spark_session
from replay.data_preparator import DataPreparator, Indexer
from replay.splitters import DateSplitter, UserSplitter
from replay.metrics import Coverage, HitRate, NDCG, MAP, Precision
from replay.experiment import Experiment

from gloolightfmwrap import LightFMGlooWrap


def main(spark: SparkSession):
    spark_conf: SparkConf = spark.sparkContext.getConf()

    if os.environ.get("PARTITION_NUM"):
        partition_num = int(os.environ.get("PARTITION_NUM"))
    else:
        if spark_conf.get("spark.cores.max") is None:
            partition_num = os.cpu_count()
        else:
            partition_num = int(spark_conf.get("spark.cores.max"))  # 28

    spark_configs = {
        "spark.driver.cores": spark_conf.get("spark.driver.cores"),
        "spark.driver.memory": spark_conf.get("spark.driver.memory"),
        "spark.memory.fraction": spark_conf.get("spark.memory.fraction"),
        "spark.executor.cores": spark_conf.get("spark.executor.cores"),
        "spark.executor.memory": spark_conf.get("spark.executor.memory"),
        "spark.executor.instances": spark_conf.get(
            "spark.executor.instances"
        ),
        "spark.sql.shuffle.partitions": spark_conf.get(
            "spark.sql.shuffle.partitions"
        ),
        "spark.default.parallelism": spark_conf.get(
            "spark.default.parallelism"
        ),
        "spark.applicationId": spark.sparkContext.applicationId,
        "seed": os.environ.get("SEED", 22),
    }

    dataset_path = os.environ.get("DATASET_PATH", "/src/ml1m_ratings.csv")
    # data = MovieLens(dataset_size)
    data = spark.read.csv(dataset_path, header=True)

    data = data\
        .withColumn("user_id", data["user_id"].cast(IntegerType()))\
        .withColumn("item_id", data["item_id"].cast(IntegerType()))\
        .withColumn('timestamp', sf.col('timestamp').cast('long'))

    preparator = DataPreparator()

    preparator.setColumnsMapping(
        {
            "user_id": "user_id",
            "item_id": "item_id",
            "relevance": "rating",
            "timestamp": "timestamp",
        }
    )
    log = preparator.transform(data).withColumnRenamed("user_id", "user_idx").withColumnRenamed("item_id", "item_idx")
    log = log.repartition(partition_num).cache()
    log.write.mode('overwrite').format('noop').save()
    only_positives_log = log.filter(sf.col('relevance') >= 3)
    train_spl = DateSplitter(
        test_start=0.2,
        drop_cold_items=True,
        drop_cold_users=True,
    )

    train, test = train_spl.split(only_positives_log)
    train = train.withColumn('relevance', sf.lit(1))
    test = test.withColumn('relevance', sf.lit(1))

    train = train.cache()
    test = test.cache()
    train.write.mode('overwrite').format('noop').save()
    test.write.mode('overwrite').format('noop').save()

    prediction_quality = Experiment(
        test,
        {
            NDCG(): [5, 10, 25, 100, 500, 1000],
            MAP(): [5, 10, 25, 100, 500, 1000],
            HitRate(): [5, 10, 25, 100, 500, 1000],
            Precision(): [5, 10, 25, 100, 500, 1000]
        },
    )

    train_repartitioned = train.repartition(partition_num, "user_idx")
    model = LightFM(loss='warp', random_state=int(spark_configs.get('seed', 22)), max_sampled=100, learning_rate=0.05)
    lightfm_wrap = LightFMGlooWrap(
        model=model,
        world_size=partition_num,
        use_spark=True,
    )

    wrapper = lightfm_wrap.fit_partial(train_repartitioned, verbose=True, epochs=30)

    test_pairs = test.select('user_idx').distinct().crossJoin(train.select('item_idx').distinct())
    filtered_test = test_pairs.join(
        train,
        [test_pairs.user_idx == train.user_idx, test_pairs.item_idx == train.item_idx],
        "leftanti"
    )
    filtered_test_pd = filtered_test.toPandas()
    filtered_test_pd["relevance"] = wrapper.model.predict(
        user_ids=filtered_test_pd["user_idx"].to_numpy(),
        item_ids=filtered_test_pd["item_idx"].to_numpy(),
        num_threads=40,
    )
    prediction_quality.add_result('wrapper_res', filtered_test_pd)
    prediction_quality.results.to_csv(
        os.path.join(os.environ.get('PATH_TO_SAVE_RESULTS', '.'), f'evaluation_results.csv')
    )


if __name__ == "__main__":
    spark_sess = get_spark_session()
    main(spark=spark_sess)
    spark_sess.stop()
