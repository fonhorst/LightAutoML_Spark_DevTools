import argparse
import json
import time

import pandas as pd
from lightfm import LightFM
from ligthfm_mpi_wrap import LightFMMPIWrap
from pyspark.sql import SparkSession, Window
from replay.experiment import Experiment
from replay.metrics import MAP, NDCG, Coverage, HitRate, Precision
from replay.session_handler import State
from scipy import sparse


def exec_time(start, end):
    diff_time = end - start
    return round(diff_time, 4)


def main(args):
    MAX_SAMPLES = 100
    NUM_THREADS = int(40 / args.num_mpi_workers)

    model = LightFM(
        loss=args.loss,
        random_state=args.random_seed,
        max_sampled=MAX_SAMPLES,
        learning_rate=0.05,
    )

    lightfm_mpi = LightFMMPIWrap(model)
    rank = lightfm_mpi.mpi_world.Get_rank()

    if rank == 0:
        spark_sess = (
            SparkSession.builder.master("local[6]")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.kryoserializer.buffer.max", "512m")
            .config("spark.driver.memory", "64g")
            .config("spark.executor.memory", "64g")
            .config("spark.sql.shuffle.partitions", "18")
            .config("spark.default.parallelism", "18")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.local.dir", "/tmp")
            .getOrCreate()
        )

        spark = State(spark_sess).session
        spark.sparkContext.setLogLevel("ERROR")

    timings = {}

    train_matrix = sparse.load_npz(args.path_to_train_interactions_csr)
    test_matrix = sparse.load_npz(args.path_to_test_interactions_csr)
    prediction_pairs = pd.read_csv(args.path_to_prediction_user_item_pairs)

    if rank == 0:
        data_test = pd.DataFrame(
            {
                "user_idx": test_matrix.tocoo().row,
                "item_idx": test_matrix.tocoo().col,
                "relevance": test_matrix.tocoo().data,
            }
        )
        prediction_quality = Experiment(
            data_test,
            {
                NDCG(): [5, 10, 25, 100, 500, 1000],
                MAP(): [5, 10, 25, 100, 500, 1000],
                HitRate(): [5, 10, 25, 100, 500, 1000],
                Precision(): [5, 10, 25, 100, 500, 1000],
            },
        )

    time_start = time.time()
    lightfm_mpi.fit_partial(
        train_matrix, epochs=args.num_epochs, num_threads=NUM_THREADS, verbose=True
    )
    time_end = time.time()

    if rank == 0:
        timings[f"training-{rank}"] = exec_time(time_start, time_end)
    print("TIME", exec_time(time_start, time_end))

    if rank == 0:
        time_predict_start = time.time()
        predict_test = lightfm_mpi.model.predict(
            user_ids=prediction_pairs.user_idx.to_numpy(),
            item_ids=prediction_pairs.item_idx.to_numpy(),
            num_threads=NUM_THREADS,
        )
        time_predict_end = time.time()
        timings[f"prediction-{rank}"] = exec_time(time_predict_start, time_predict_end)

        prediction_pairs["relevance"] = predict_test

        time_start_metrics = time.time()
        prediction_quality.add_result(
            f"MPI-LightFM-{args.dataset_name}-N-{args.num_mpi_workers}-M-{NUM_THREADS}",
            prediction_pairs,
        )
        time_end_metrics = time.time()
        print(exec_time(time_start_metrics, time_end_metrics))
        timings[f"metrics_calc-{rank}"] = exec_time(
            time_start_metrics, time_end_metrics
        )

        prediction_quality.results.to_csv(
            f"full_res_mpi/bpr-warp/{args.num_epochs}epochs-{args.dataset_name}/{args.loss}-lightfm-{args.num_mpi_workers}-{args.dataset_name}.csv"
        )

    with open(
        f"full_res_mpi/bpr-warp/{args.num_epochs}epochs-{args.dataset_name}/{args.loss}-{args.num_mpi_workers}_MPI-{args.dataset_name}.json",
        "w",
    ) as f:
        json.dump(timings, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--loss",
        default=None,
        type=str,
        required=True,
        help="Loss: one of `bpr`, `warp`",
    )
    parser.add_argument(
        "--num_mpi_workers",
        default=None,
        type=int,
        required=True,
        help="Number of MPI wokers",
    )
    parser.add_argument(
        "--path_to_train_interactions_csr",
        default=None,
        type=str,
        required=True,
        help="Path to train interactions matrix in CSR scipy format",
    )
    parser.add_argument(
        "--path_to_test_interactions_csr",
        default=None,
        type=str,
        required=True,
        help="Path to test interactions matrix in CSR scipy format",
    )
    parser.add_argument(
        "--path_to_prediction_user_item_pairs",
        default=None,
        type=str,
        required=True,
        help="Path to prediction pairs CSV",
    )
    parser.add_argument(
        "--random_seed", default=22, type=int, required=False, help="Random seed",
    )
    parser.add_argument(
        "--dataset_name",
        default=None,
        type=str,
        required=True,
        help="Name of the dataset, its size or any identifier for the results files",
    )
    parser.add_argument(
        "--num_epochs",
        default=30,
        type=int,
        required=False,
        help="Number of epochs to run",
    )

    args = parser.parse_args()
    main(args)
