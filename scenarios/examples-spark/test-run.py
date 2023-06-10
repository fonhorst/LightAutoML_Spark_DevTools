import logging
from logging import config

from pyspark.sql import SparkSession
from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT, log_exec_timer

# from examples_utils import get_spark_session

logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename='/tmp/slama.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)

with log_exec_timer("mlflow_import") as timer:
    import mlflow

print(f"Timer mlflow: {timer.duration}")


if __name__ == "__main__":
    logger.info("In the very beginning of parallel-optuna")
    spark = SparkSession.builder.getOrCreate()

    print(spark.sparkContext.parallelize(list(range(10))).sum())

    spark.stop()
