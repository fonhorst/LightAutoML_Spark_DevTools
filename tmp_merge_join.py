from contextlib import contextmanager
from random import random

from pandas import Series
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.pandas.functions import pandas_udf
from pyspark.sql.types import IntegerType
from pyspark.sql import DataFrame as SparkDataFrame, Column


@contextmanager
def JobGroup(group_id: str, description: str):
    sc = SparkSession.getActiveSession().sparkContext
    sc.setJobGroup(group_id, description)
    yield
    sc._jsc.clearJobGroup()


spark = (
    SparkSession
    .builder
    .master('local-cluster[4,1,1024]')
    .config('spark.sql.autoBroadcastJoinThreshold', '-1')
    .getOrCreate()
)

BUCKET_NUMS = 10
ID_COL = 'id'

def bucketize_table(name: str, df: SparkDataFrame) -> SparkDataFrame:
    with JobGroup("bucketize table", f"Bucketizing table {name}"):
        (
            df
            .repartition(BUCKET_NUMS, ID_COL)
            .write
            .mode('overwrite')
            .bucketBy(BUCKET_NUMS, ID_COL).sortBy(ID_COL)
            .saveAsTable(name, format='parquet', path=f"/tmp/{name}")
        )
        df_bucketed = spark.table(name)
    return df_bucketed

data_1 = [{"id": i, "b": i + 1, "c": str(i*10)} for i in range(10_000)]

data_2 = [{"id": i, "d": i + 2, "e": str(i + 10)} for i in range(10_000)]

df_1 = spark.createDataFrame(data_1)
df_2 = spark.createDataFrame(data_2)

# (
#     df_1.repartition(10, 'id').write.mode('overwrite')
#     .bucketBy(10, 'id').sortBy('id')
#     .saveAsTable('df_1_bucketed', format='parquet', path="/tmp/data_1_bucketed")
# )
# df_1_bucketed = spark.table('df_1_bucketed')

df_1_bucketed = bucketize_table('df_1_bucketed', df_1)

# (
#     df_2.repartition(10, 'id').write.mode('overwrite')
#     .bucketBy(10, 'id').sortBy('id')
#     .saveAsTable('df_2_bucketed', format='parquet', path="/tmp/data_2_bucketed")
# )
# df_2_bucketed = spark.table('df_2_bucketed')

df_2_bucketed = bucketize_table('df_2_bucketed', df_2)


df_1_bucketed.write.parquet("file:///tmp/df_1_bucketed.parquet", mode='overwrite')
df_2_bucketed.write.parquet("file:///tmp/df_2_bucketed.parquet", mode='overwrite')

# Exchange + ZippedPartitions
# df_1.join(df_2, on='id', how='inner').write.mode('overwrite').format('noop').save()
# merge join with ZippedPartitions
joined_df = df_1_bucketed.join(df_2_bucketed, on='id', how='inner')
joined_df.write.parquet("file:///tmp/test.parquet", mode='overwrite')



# The following part may be executed in a separate Spark App without the first part
# if the first part has been already persisted
# spark.sql("""
#     CREATE TABLE loaded_bucketed_1
#     (id long, d long, e string)
#     USING PARQUET
#     CLUSTERED BY (id) SORTED BY (id) INTO 10 BUCKETS
#     LOCATION '/tmp/data_2_bucketed'
#     """.strip())
# df_3_bucketed = spark.table("loaded_bucketed_1")
#
# spark.sql("""
#     CREATE TABLE loaded_bucketed_2
#     (id long, d long, e string)
#     USING PARQUET
#     CLUSTERED BY (id) SORTED BY (id) INTO 10 BUCKETS
#     LOCATION '/tmp/data_2_bucketed'
#     """.strip())
# df_4_bucketed = spark.table("loaded_bucketed_2")
#
# with JobGroup("3_4_bucketed_join", "Bucketed join on loaded tables"):
#     # merge join with ZippedPartitions (no merge join with sorting aka SortMergeJoin without Exchange )
#     df_3_bucketed.join(df_4_bucketed, on='id', how='inner').write.mode('overwrite').format('noop').save()
#
# temp_df = df_4_bucketed.where(F.col('d') % 2 == 0).cache()
# temp_df.write.mode('overwrite').format('noop').save()
# # merge join with ZippedPartitions
# df_3_bucketed.join(temp_df, on='id', how='inner').write.mode('overwrite').format('noop').save()
#
#
# # @pandas_udf("float")
# # def func_rand(s: Series) -> Series:
# #     return s.map(lambda x: random())
# random_udf = udf(lambda: int(random() * 100), IntegerType())
#
# even_df = df_4_bucketed.where(F.col('d') % 2 == 0).withColumn("ccc", random_udf()).cache()
# even_df.write.mode('overwrite').format('noop').save()
#
# odd_df = df_4_bucketed.where(F.col('d') % 2 == 1).cache()
# odd_df.write.mode('overwrite').format('noop').save()
#
# # exchange, no bucketing
# full_df = temp_df.unionByName(odd_df)
# df_3_bucketed.join(full_df, on='id', how='inner').write.mode('overwrite').format('noop').save()
#
# with JobGroup("even_odd_join", "even odd join example"):
#     # all merge joins with ZippedPartitions
#     (
#         df_3_bucketed
#         .join(even_df, on='id', how='inner')
#         .join(odd_df, on='id', how='inner')
#         .write.mode('overwrite').format('noop').save()
#     )
#
# spark.stop()