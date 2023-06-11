import itertools

parallelism_degrees = ["1", "2", "4", "8", "16"]

datasets = [
    "hdfs://hacluster:8020/tmp/spark_data/feature_processor/kaggle-used-cars__lgb_adv.dataset",
    "hdfs://hacluster:8020/tmp/spark_data/openml_datasets/Airlines_DepDelay_1M__linear.csv"
]

configurations = {
    "path_to_save_params": "/tmp/experimental_parameters",
    "configuration": [
        {
            "cmd": "bash",
            "experiment_script_path": "/opt/experiments/scripts/experiment.py",
            "spark_submit_exec_path": "/src/repeater/optuna-spark-submit",
            "workdir": "/src/repeater",
            "mlflow_experiment_id": 167,
            "env_parameters": {
                "DATASET": datset,
                "EXP_JOB_PARALLELISM": parallelism,
                "DRIVER_CORES": "6",
                "DRIVER_MEMORY": "16g",
                "DRIVER_MAX_RESULT_SIZE": "5g",
                "EXECUTOR_INSTANCES": "16",
                "EXECUTOR_CORES": "4",
                "EXECUTOR_MEMORY": "40g"
            },
            "run_parameters": {
                "feat_pipe": "lgb_adv",
                "n_trials": 64,
                "timeout": 60000,
                "stabilize": 0,
                "numIterations": 500,
                "earlyStoppingRound": 50000
            }
        }
        for parallelism, datset in itertools.product(parallelism_degrees, datasets)
    ]
}
