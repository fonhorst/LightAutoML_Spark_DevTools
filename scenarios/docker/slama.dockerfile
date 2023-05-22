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

ARG SLAMA_WHEEL_VERSION=0.3.2
ARG SLAMA_JAR_VERSION=0.1.1

COPY $SLAMA_BUILD_TMP/requirements.txt /src
COPY $SLAMA_BUILD_TMP/sparklightautoml_dev-${SLAMA_WHEEL_VERSION}-py3-none-any.whl /src/SparkLightAutoML-${SLAMA_WHEEL_VERSION}-py3-none-any.whl
COPY $SLAMA_BUILD_TMP/spark-lightautoml_2.12-${SLAMA_JAR_VERSION}.jar /src
COPY $SLAMA_BUILD_TMP/examples-spark /src/examples-spark

ENV SLAMA_WHEEL_VERSION=${SLAMA_WHEEL_VERSION}
ENV SLAMA_JAR_VERSION=${SLAMA_JAR_VERSION}

WORKDIR /src

ENTRYPOINT ["/src/yarn-submit"]
