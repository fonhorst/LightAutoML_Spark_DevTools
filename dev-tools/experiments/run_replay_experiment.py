import os
import logging.config

import mlflow
import pandas as pd
from pyspark.sql import SparkSession
from replay.data_preparator import DataPreparator, Indexer
from replay.session_handler import get_spark_session
from replay.utils import JobGroup, log_exec_timer

from replay.models import (
    ALSWrap,
    SLIM,
    LightFMWrap,
    ItemKNN,
    Word2VecRec,
    PopRec,
    RandomRec,
    AssociationRulesItemRec,
)

from rs_datasets import MovieLens, MillionSongDataset
from pyspark.sql import functions as sf
from replay.splitters import DateSplitter, UserSplitter
from replay.utils import get_log_info


VERBOSE_LOGGING_FORMAT = (
    "%(asctime)s %(levelname)s %(module)s %(filename)s:%(lineno)d %(message)s"
)
logging.basicConfig(level=logging.INFO, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger("replay")
logger.setLevel(logging.DEBUG)


def main(spark: SparkSession, dataset_name: str):
    K = 10
    SEED = int(os.getenv("SEED"))  # 1234
    n_samples = None
    MLFLOW_TRACKING_URI = os.getenv(
        "MLFLOW_TRACKING_URI"
    )  # "http://node2.bdcl:8811"
    MODEL = os.getenv("MODEL")  # "ALS"
    partition_num = int(
        spark.sparkContext.getConf().get("spark.cores.max")
    )  # 28

    if dataset_name.startswith("MovieLens"):
        # name__size__sample pattern
        dataset_params = dataset_name.split("__")
        if len(dataset_params) == 1:
            data = MovieLens(
                "1m", path="/opt/spark_data/replay_datasets/MovieLens"
            )
        elif len(dataset_params) == 2:
            data = MovieLens(
                dataset_params[1],
                path="/opt/spark_data/replay_datasets/MovieLens",
            )
        elif len(dataset_params) == 3:
            n_samples = int(dataset_params[2])
            data = MovieLens(
                dataset_params[1],
                path="/opt/spark_data/replay_datasets/MovieLens",
            )
        else:
            raise ValueError("Too many dataset params.")
        data = data.ratings
        mapping = {
            "user_id": "user_id",
            "item_id": "item_id",
            "relevance": "rating",
            "timestamp": "timestamp",
        }
    elif dataset_name.startswith("MillionSongDataset"):
        # name__sample pattern
        dataset_params = dataset_name.split("__")
        if len(dataset_params) == 2:
            n_samples = int(dataset_params[1])

        data = MillionSongDataset(
            path="/opt/spark_data/replay_datasets/MillionSongDataset"
        )
        data = pd.concat([data.train, data.test, data.val])
        mapping = {
            "user_id": "user_id",
            "item_id": "item_id",
            "relevance": "play_count",
        }
    else:
        raise ValueError("Unknown dataset.")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(f"replay-{MODEL}")

    with mlflow.start_run():

        mlflow.log_param(
            "spark.driver.cores",
            spark.sparkContext.getConf().get("spark.driver.cores"),
        )
        mlflow.log_param(
            "spark.driver.memory",
            spark.sparkContext.getConf().get("spark.driver.memory"),
        )
        mlflow.log_param(
            "spark.memory.fraction",
            spark.sparkContext.getConf().get("spark.memory.fraction"),
        )
        mlflow.log_param(
            "spark.executor.cores",
            spark.sparkContext.getConf().get("spark.executor.cores"),
        )
        mlflow.log_param(
            "spark.executor.memory",
            spark.sparkContext.getConf().get("spark.executor.memory"),
        )
        mlflow.log_param(
            "spark.executor.instances",
            spark.sparkContext.getConf().get("spark.executor.instances"),
        )
        mlflow.log_param(
            "spark.sql.shuffle.partitions",
            spark.sparkContext.getConf().get("spark.sql.shuffle.partitions"),
        )
        mlflow.log_param(
            "spark.default.parallelism",
            spark.sparkContext.getConf().get("spark.default.parallelism"),
        )
        mlflow.log_param(
            "spark.applicationId", spark.sparkContext.applicationId
        )

        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("seed", SEED)
        mlflow.log_param("K", K)

        if n_samples:
            with log_exec_timer("Dataset sampling") as dataset_sampling_timer:
                data = data.sample(n=n_samples, random_state=SEED)
            mlflow.log_metric(
                "dataset_sampling_sec", dataset_sampling_timer.duration
            )

        with log_exec_timer("DataPreparator execution") as preparator_timer:
            preparator = DataPreparator()
            log = preparator.transform(columns_mapping=mapping, data=data)
            log = log.repartition(partition_num).cache()
            log.write.mode("overwrite").format("noop").save()
        mlflow.log_metric("preparator_sec", preparator_timer.duration)

        mlflow.log_metric("log_num_partitions", log.rdd.getNumPartitions())

        with log_exec_timer("log filtering") as log_filtering_timer:
            # will consider ratings >= 3 as positive feedback. A positive feedback is treated with relevance = 1
            only_positives_log = log.filter(
                sf.col("relevance") >= 3
            ).withColumn("relevance", sf.lit(1))
            only_positives_log = only_positives_log.cache()
            only_positives_log.write.mode("overwrite").format("noop").save()
        mlflow.log_metric("log_filtering_sec", log_filtering_timer.duration)

        with log_exec_timer(
            "only_positives_log.count() execution"
        ) as log_count_timer:
            log_length = only_positives_log.count()
        mlflow.log_metric("log_count_sec", log_count_timer.duration)
        mlflow.log_param("log_length", log_length)

        with log_exec_timer("Indexer training") as indexer_fit_timer:
            indexer = Indexer(user_col="user_id", item_col="item_id")
            indexer.fit(
                users=log.select("user_id"), items=log.select("item_id")
            )
        mlflow.log_metric("indexer_fit_sec", indexer_fit_timer.duration)

        with log_exec_timer("Indexer transform") as indexer_transform_timer:
            log_replay = indexer.transform(df=only_positives_log)
            log_replay = log_replay.cache()
            log_replay.write.mode("overwrite").format("noop").save()
        mlflow.log_metric(
            "indexer_transform_sec", indexer_transform_timer.duration
        )

        with log_exec_timer("DateSplitter execution") as splitter_timer:
            ## MovieLens
            # train_spl = DateSplitter(
            #     test_start=0.2,
            #     drop_cold_items=True,
            #     drop_cold_users=True,
            # )

            ## MillionSongDataset
            train_spl = UserSplitter(
                item_test_size=0.2,
                shuffle=True,
                drop_cold_items=True,
                drop_cold_users=True,
            )
            train, test = train_spl.split(log_replay)

            train = train.cache()
            test = test.repartition(partition_num).cache()
            train.write.mode("overwrite").format("noop").save()
            test.write.mode("overwrite").format("noop").save()
        mlflow.log_metric("splitter_sec", splitter_timer.duration)

        mlflow.log_metric("train_num_partitions", train.rdd.getNumPartitions())
        mlflow.log_metric("test_num_partitions", test.rdd.getNumPartitions())

        # with log_exec_timer("get_log_info() execution") as get_log_info_timer:
        #     train_info = get_log_info(train)
        #     test_info = get_log_info(test)
        #     logger.info(f'train info: {train_info}')
        #     logger.info(f'test info: {test_info}')
        # mlflow.log_metric("get_log_info_sec", get_log_info_timer.duration)
        # mlflow.log_param("train_info", train_info)
        # mlflow.log_param("test_info", test_info)

        with log_exec_timer(
            "train.count() and test.count() execution"
        ) as train_test_count_timer:
            test_size = test.count()
            train_size = train.count()
        mlflow.log_metric(
            "train_test_count_sec", train_test_count_timer.duration
        )
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("train_size", train_size)

        mlflow.log_param("model", MODEL)
        if MODEL == "ALS":
            # ALS_RANK = 100
            # model = ALSWrap(rank=ALS_RANK, seed=SEED)
            # mlflow.log_param("ALS_rank", ALS_RANK)
            model = ALSWrap(
                seed=SEED,
                num_item_blocks=partition_num,
                num_user_blocks=partition_num,
            )
        elif MODEL == "Explicit_ALS":
            model = ALSWrap(seed=SEED, implicit_prefs=False)
        elif MODEL == "SLIM":
            model = SLIM(seed=SEED)
        elif MODEL == "ItemKNN":
            model = ItemKNN()
        elif MODEL == "LightFM":
            model = LightFMWrap(random_state=SEED)
        elif MODEL == "Word2VecRec":
            model = Word2VecRec(
                seed=SEED,
                num_partitions=partition_num,
            )
        elif MODEL == "PopRec":
            model = PopRec()
        elif MODEL == "RandomRec_uniform":
            model = RandomRec(seed=SEED, distribution="uniform")
        elif MODEL == "RandomRec_popular_based":
            model = RandomRec(seed=SEED, distribution="popular_based")
        elif MODEL == "AssociationRulesItemRec":
            model = AssociationRulesItemRec()
        else:
            raise ValueError("Unknown model.")

        with log_exec_timer(f"{MODEL} training") as train_timer, JobGroup(
            f"{model.__class__.__name__}.fit()", "Model training"
        ):
            model.fit(log=train)
        mlflow.log_metric("train_sec", train_timer.duration)

        with log_exec_timer(f"{MODEL} prediction") as infer_timer:
            recs = model.predict(
                k=K,
                users=test.select("user_idx").distinct(),
                log=train,
                filter_seen_items=True,
            )
            recs.write.mode("overwrite").format("noop").save()
        mlflow.log_metric("infer_sec", infer_timer.duration)


if __name__ == "__main__":
    spark_sess = get_spark_session()
    dataset = os.getenv("DATASET")
    # dataset = "MovieLens__1m"
    # dataset = "MillionSongDataset"
    main(spark=spark_sess, dataset_name=dataset)
    spark_sess.stop()
