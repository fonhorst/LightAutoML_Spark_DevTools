import itertools

# executors = ["1", "2", "4", "16"]
# datasets = ["higgs"]

# executors = ["16"]
# datasets = ["kaggle_a", "used_cars_dataset_1x"]

# executors = ["1", "2", "4", "8", "16"]
# datasets = ["kaggle_a",  "used_cars_dataset", "Click_prediction_small",  "covertype",  "Higgs"]


# executors = ["1", "2", "4", "8", "16"]
# columns = [10, 100, 200, 500]
## rows = ["100k", "250k", "500k", "1m"]
#
# executors = ["1", "2", "4", "8", "16"]
# columns = [10, 100, 200, 500]
# rows = ["1m"]
# datasets = [f"test_cardinality_{r}_{c}c" for r, c in itertools.product(rows, columns)]


executors = ["1", "2", "4", "8", "16"]
columns = [10, 100, 200, 500]
rows = ["100k", "250k", "500k"]
datasets = [f"test_cardinality_{r}_{c}c" for r, c in itertools.product(rows, columns)]


configurations = {
    "path_to_save_params": "/tmp/experimental_parameters",
    "configuration": [
        {
            "cmd": "bash",
            "experiment_script_path": "/src/examples-spark/scalability-feature-processing-only.py",
            "spark_submit_exec_path": "/src/yarn-submit",
            "workdir": "/src",
            "mlflow_experiment_id": "170",
            "env_parameters": {
                "HADOOP_CONF_DIR": "/etc/hadoop",
                "SLAMA_WHEEL_VERSION": "0.3.2",
                "SLAMA_JAR_VERSION": "0.1.1",
                "EXP_FEAT_PIPE": "lgb_adv",
                "DATASET": dataset,
                "DRIVER_CORES": "6",
                "DRIVER_MEMORY": "16g",
                "DRIVER_MAX_RESULT_SIZE": "5g",
                "EXECUTOR_INSTANCES": execs,
                "EXECUTOR_CORES": "4",
                "EXECUTOR_MEMORY": "40g",
                "PYSPARK_PYTHON_PATH": "/python_envs/.replay_venv/bin/python3.9",
                "WAREHOUSE_DIR": "hdfs://node21.bdcl:9000/tmp/slama-spark-warehouse-1x"
            }
        }
        for execs, dataset in itertools.product(executors, datasets)
    ]
}
