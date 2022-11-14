FROM python:3.9

ARG SLAMA_BUILD_TMP=slama_build_tmp

RUN apt-get update && \
	apt-get install -y openjdk-11-jre net-tools wget nano iputils-ping curl && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

COPY yarn-submit .

RUN chmod 555 yarn-submit

# we install only pyspark, because we need to obtain spark-submit command
RUN pip install pyspark==3.2.0
RUN ln -s /usr/local/lib/python3.9/site-packages/pyspark/bin/spark-submit /usr/bin/spark-submit

COPY yarn-submit .
RUN chmod 755 yarn-submit

COPY $SLAMA_BUILD_TMP/requirements.txt .
COPY $SLAMA_BUILD_TMP/SparkLightAutoML-0.3.0-py3-none-any.whl .
COPY $SLAMA_BUILD_TMP/spark-lightautoml_2.12-0.1.jar .

COPY $SLAMA_BUILD_TMP/examples-spark .

ENTRYPOINT ["/yarn-submit"]
