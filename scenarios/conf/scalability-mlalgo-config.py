import itertools

# executors = ["1", "2", "4", "8", "16"]
# ml_algos = ["linear_l2", "lgb"]

# executors = ["1", "2", "4", "8"]
# ml_algos = ["lgb"]

# executors = ["16"]
# ml_algos = ["linear_l2", "lgb"]

executors = ["1", "2", "4", "8"]
ml_algos = ["linear_l2", "lgb"]
datasets = ["synth_10kk_100", "synth_5kk_100"]

configurations = {
    "path_to_save_params": "/tmp/experimental_parameters",
    "configuration": [
        {
            "cmd": "bash",
            "experiment_script_path": "/src/examples-spark/scalability-ml-algos.py",
            "spark_submit_exec_path": "/src/yarn-submit",
            "workdir": "/src",
            "mlflow_experiment_id": "168",
            "env_parameters": {
                "HADOOP_CONF_DIR": "/etc/hadoop",
                "SLAMA_WHEEL_VERSION": "0.3.2",
                "SLAMA_JAR_VERSION": "0.1.1",
                "EXP_ML_ALGO": ml_algo,
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
        for ml_algo, execs, dataset in itertools.product(ml_algos, executors, datasets)
    ]
}
