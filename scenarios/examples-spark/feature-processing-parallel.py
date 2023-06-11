import logging.config
import os

from pyspark.sql import SparkSession
from sparklightautoml.computations.parallel import ParallelComputationsManager
from sparklightautoml.pipelines.features.base import SparkFeaturesPipeline
from sparklightautoml.pipelines.features.lgb_pipeline import SparkLGBAdvancedPipeline, SparkLGBSimpleFeatures
from sparklightautoml.pipelines.features.linear_pipeline import SparkLinearFeatures
from sparklightautoml.reader.base import SparkToSparkReader
from sparklightautoml.tasks.base import SparkTask
from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT, log_exec_timer

from examples_utils import get_dataset, initialize_environment

logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename='/tmp/slama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


feature_pipelines = {
    "linear": SparkLinearFeatures(),
    "lgb_simple": SparkLGBSimpleFeatures(),
    "lgb_adv": SparkLGBAdvancedPipeline()
}


@initialize_environment
def main(spark: SparkSession):
    # settings and data
    cv = 5
    job_parallelism = int(os.environ.get("EXP_JOB_PARALLELISM", "1"))
    dataset_name = os.environ.get("DATASET", "lama_test_dataset")

    dataset = get_dataset(dataset_name)
    df = dataset.load()

    # mlflow.log_params({
    #     "cv": cv,
    #     "dataset": dataset_name,
    #     "job_parallelism": job_parallelism,
    #     "dataset_path": dataset.path
    # })

    computations_manager = ParallelComputationsManager(parallelism=job_parallelism)
    task = SparkTask(name=dataset.task_type)
    reader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False)

    ds = reader.fit_read(train_data=df, roles=dataset.roles)

    def build_task(fp_name: str, feature_pipe: SparkFeaturesPipeline):
        def func():
            logger.info(f"Calculating feature pipeline: {fp_name}")
            with log_exec_timer(f"{fp_name}_pipe_fit") as timer:
                fp_ds = feature_pipe.fit_transform(ds)
                save_path = f"file:///opt/spark_data/preproccessed_datasets/{dataset_name}__{fp_name}__features.dataset"
                fp_ds.save(save_path, save_mode='overwrite')

            # mlflow.log_metric(timer.name, timer.duration)

            logger.info(f"Finished calculating pipeline: {fp_name}")

        return func

    tasks = [build_task(feat_pipe_name, feature_pipe) for feat_pipe_name, feature_pipe in feature_pipelines.items()]

    with log_exec_timer("feat_pipes_fit") as timer:
        computations_manager.compute(tasks)

    # mlflow.log_metric(timer.name, timer.duration)


if __name__ == "__main__":
    main()
