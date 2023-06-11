import itertools

parallelism_degrees = ["1", "2", "4"]#, "8", "16"]

datasets = [
    # "synth_10kk_100",
    # "synth_5kk_100"
    "lama_test_dataset"
]

configurations = {
    "path_to_save_params": "/tmp/experimental_parameters",
    "configuration": [
        {
            "cmd": "bash",
            "experiment_script_path": "/src/examples-spark/parallel-optuna.py",
            "spark_submit_exec_path": "/src/yarn-submit",
            "workdir": "/src",
            "mlflow_experiment_id": "167",
            "env_parameters": {
                "HADOOP_CONF_DIR": "/etc/hadoop",
                "SLAMA_WHEEL_VERSION": "0.3.2",
                "SLAMA_JAR_VERSION": "0.1.1",
                "DATASET": datset,
                "EXP_JOB_PARALLELISM": parallelism,
                "DRIVER_CORES": "6",
                "DRIVER_MEMORY": "16g",
                "DRIVER_MAX_RESULT_SIZE": "5g",
                "EXECUTOR_INSTANCES": "4",
                "EXECUTOR_CORES": "4",
                "EXECUTOR_MEMORY": "40g",
                "PYSPARK_PYTHON_PATH": "/python_envs/.replay_venv/bin/python3.9",
                "WAREHOUSE_DIR": "hdfs://node21.bdcl:9000/tmp/slama-spark-warehouse-1x"
            },
            # "run_parameters": {
            #     "feat_pipe": "lgb_adv",
            #     "n_trials": 64,
            #     "timeout": 60000,
            #     "stabilize": 0,
            #     "numIterations": 500,
            #     "earlyStoppingRound": 50000
            # }
        }
        for parallelism, datset in itertools.product(parallelism_degrees, datasets)
    ]
}
