ARG IMAGE_TAG=slama-yarn
FROM node2.bdcl:5000/yarn-base:${IMAGE_TAG}

HEALTHCHECK CMD curl -f http://localhost:8042/ || exit 1

ADD run.sh /run.sh
RUN chmod a+x /run.sh

EXPOSE 8042

RUN apt update

RUN apt install build-essential libncurses5-dev zlib1g-dev libnss3-dev libgdbm-dev libssl-dev libsqlite3-dev libffi-dev libreadline-dev curl libbz2-dev wget -y

# Python 3.8
RUN wget https://www.python.org/ftp/python/3.8.13/Python-3.8.13.tgz

RUN tar -xvf Python-3.8.13.tgz

RUN rm Python-3.8.13.tgz

RUN cd Python-3.8.13 && ./configure --enable-optimizations

RUN cd Python-3.8.13 && make altinstall

RUN ln -s /usr/local/bin/python3.8 /usr/bin/python3.8

# Python 3.9
RUN wget https://www.python.org/ftp/python/3.9.12/Python-3.9.12.tgz

RUN tar -xvf Python-3.9.12.tgz

RUN rm Python-3.9.12.tgz

RUN cd Python-3.9.12 && ./configure --enable-optimizations

RUN cd Python-3.9.12 && make altinstall

RUN ln -s /usr/local/bin/python3.9 /usr/bin/python3.9

# install lama deps
COPY requirements-3.9.txt ./

RUN pip3.9 install --upgrade pip && pip3.9 install -r requirements-3.9.txt --use-deprecated=legacy-resolver

RUN python3.9 -c 'from pyspark.sql import SparkSession; SparkSession.builder.config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.5").config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven").getOrCreate()'

RUN mv /root/.ivy2/jars /usr/local/lib/synapseml_0.9.5_jars

RUN python3.9 -m pip install lightautoml==0.3.7.1 mlflow-skinny


# pyspark 3.0.1
COPY requirements-3.8-spark-3.0.1.txt ./

RUN pip3.8 install --upgrade pip && pip3.8 install -r requirements-3.8-spark-3.0.1.txt --use-deprecated=legacy-resolver

RUN mkdir /root/.ivy2/jars/

RUN wget https://repo1.maven.org/maven2/io/netty/netty-transport-native-epoll/4.1.68.Final/netty-transport-native-epoll-4.1.68.Final.jar

RUN mv netty-transport-native-epoll-4.1.68.Final.jar /root/.ivy2/jars/io.netty_netty-transport-native-epoll-4.1.68.Final.jar

RUN wget https://repo1.maven.org/maven2/io/netty/netty-transport-native-kqueue/4.1.68.Final/netty-transport-native-kqueue-4.1.68.Final.jar

RUN mv netty-transport-native-kqueue-4.1.68.Final.jar /root/.ivy2/jars/io.netty_netty-transport-native-kqueue-4.1.68.Final.jar

RUN wget https://repo1.maven.org/maven2/io/netty/netty-resolver-dns-native-macos/4.1.68.Final/netty-resolver-dns-native-macos-4.1.68.Final.jar

RUN mv netty-resolver-dns-native-macos-4.1.68.Final.jar /root/.ivy2/jars/io.netty_netty-resolver-dns-native-macos-4.1.68.Final.jar

RUN python3.8 -c 'from pyspark.sql import SparkSession; SparkSession.builder.config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.5").config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven").getOrCreate()'

RUN mv /root/.ivy2/jars /usr/local/lib/spark_3.0.1_synapseml_0.9.5_jars

RUN rm /usr/local/lib/spark_3.0.1_synapseml_0.9.5_jars/com.fasterxml.jackson*

RUN python3.8 -m pip install lightautoml==0.3.7.1 mlflow-skinny


# pyspark 3.1.1, synapseml 0.9.5
RUN python3.9 -m venv ./.venv

COPY requirements-3.9-spark-3.1.1.txt ./

RUN ./.venv/bin/python3.9 -m pip install -r requirements-3.9-spark-3.1.1.txt --use-deprecated=legacy-resolver && ./.venv/bin/python3.9 -m pip install lightautoml==0.3.7.1 mlflow-skinny

RUN ./.venv/bin/python3.9 -c 'from pyspark.sql import SparkSession; SparkSession.builder.config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.5-35-e962330b-SNAPSHOT").config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven").getOrCreate()'

RUN mv /root/.ivy2/jars /usr/local/lib/spark_3.1.1_synapseml_0.9.5_jars

# RUN rm /usr/local/lib/spark_3.1.1_synapseml_0.9.5_jars/com.fasterxml.jackson*
# ============


# pyspark 3.1.1, synapseml 0.9.4
RUN python3.9 -m venv ./.venv_3.1.1_0.9.4

COPY requirements-3.9-spark-3.1.1-synapseml-0.9.4.txt ./

RUN ./.venv_3.1.1_0.9.4/bin/python3.9 -m pip install -r requirements-3.9-spark-3.1.1-synapseml-0.9.4.txt --use-deprecated=legacy-resolver && ./.venv_3.1.1_0.9.4/bin/python3.9 -m pip install lightautoml==0.3.7.1 mlflow-skinny

RUN ./.venv_3.1.1_0.9.4/bin/python3.9 -c 'from pyspark.sql import SparkSession; SparkSession.builder.config("spark.jars.packages", "com.microsoft.azure:synapseml_2.12:0.9.4").config("spark.jars.repositories", "https://mmlspark.azureedge.net/maven").getOrCreate()'

RUN mv /root/.ivy2/jars /usr/local/lib/spark_3.1.1_synapseml_0.9.4_jars


CMD ["/run.sh"]
