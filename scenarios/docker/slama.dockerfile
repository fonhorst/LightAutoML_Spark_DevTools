FROM python:3.9

ARG SLAMA_BUILD_TMP=slama_build_tmp

RUN apt-get update && \
	apt-get install -y openjdk-11-jre net-tools wget nano iputils-ping curl && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

# we install only pyspark, because we need to obtain spark-submit command
RUN pip install pyspark==3.2.0
RUN ln -s /usr/local/lib/python3.9/site-packages/pyspark/bin/spark-submit /usr/bin/spark-submit
RUN mkdir -p /src

COPY yarn-submit /src
RUN chmod 755 /src/yarn-submit

COPY tabular_config.yml /src

COPY $SLAMA_BUILD_TMP/requirements.txt /src
COPY $SLAMA_BUILD_TMP/SparkLightAutoML_DEV-0.3.2-py3-none-any.whl /src/SparkLightAutoML-0.3.2-py3-none-any.whl
COPY $SLAMA_BUILD_TMP/spark-lightautoml_2.12-0.1.1.jar /src

COPY $SLAMA_BUILD_TMP/examples-spark /src/examples-spark

WORKDIR /src

ENTRYPOINT ["/src/yarn-submit"]
