import itertools

executors = ["1", "2", "4", "8"]
# datasets = ["used_cars_dataset_1x", "kaggle_a"]
datasets = ["kaggle_a"]
# datasets = ["lama_test_dataset"]

configurations = {
    "path_to_save_params": "/tmp/experimental_parameters",
    "configuration": [
        {
            "cmd": "bash",
            "experiment_script_path": "/src/examples-spark/cat-boost.py",
            "spark_submit_exec_path": "/src/cb-yarn-submit",
            "workdir": "/src",
            "mlflow_experiment_id": "168",
            "env_parameters": {
                "HADOOP_CONF_DIR": "/etc/hadoop",
                "SLAMA_WHEEL_VERSION": "0.3.2",
                "SLAMA_JAR_VERSION": "0.1.1",
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
