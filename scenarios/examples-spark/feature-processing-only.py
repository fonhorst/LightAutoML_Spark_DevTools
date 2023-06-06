import logging.config
import uuid
from collections import Counter

from sparklightautoml.utils import logging_config, VERBOSE_LOGGING_FORMAT

from examples_utils import get_spark_session, get_dataset
from sparklightautoml.pipelines.features.lgb_pipeline import SparkLGBAdvancedPipeline, SparkLGBSimpleFeatures
from sparklightautoml.pipelines.features.linear_pipeline import SparkLinearFeatures
from sparklightautoml.reader.base import SparkToSparkReader
from sparklightautoml.tasks.base import SparkTask
import os

uid = uuid.uuid4()
logging.config.dictConfig(logging_config(level=logging.DEBUG, log_filename=f'/tmp/slama-{uid}.log'))
logging.basicConfig(level=logging.DEBUG, format=VERBOSE_LOGGING_FORMAT)
logger = logging.getLogger(__name__)


feature_pipelines = {
    "linear": SparkLinearFeatures(),
    "lgb_simple": SparkLGBSimpleFeatures(),
    "lgb_adv": SparkLGBAdvancedPipeline()
}


if __name__ == "__main__":
    spark = get_spark_session()

    # settings and data
    # params
    cv = 5
    feat_pipe = "lgb_adv"  # linear, lgb_simple or lgb_adv
    dataset_name = os.environ.get("DATASET", "lama_test_dataset")
    dataset = get_dataset(dataset_name)
    df = dataset.load()

    # params: count rows + columns

    task = SparkTask(name=dataset.task_type)
    reader = SparkToSparkReader(task=task, cv=cv, advanced_roles=False)
    feature_pipe = feature_pipelines.get(feat_pipe, None)

    assert feature_pipe, f"Unsupported feat pipe {feat_pipe}"

    # 1
    ds = reader.fit_read(train_data=df, roles=dataset.roles)

    # params
    Counter(type(role).__name__ for feat, role in ds.roles.items())

    # 2
    ds = feature_pipe.fit_transform(ds)

    # 3
    # save processed data
    # param
    save_path = f"file:///opt/spark_data/preproccessed_datasets/{dataset_name}__{feat_pipe}__features.dataset"
    ds.save(save_path, save_mode='overwrite')
