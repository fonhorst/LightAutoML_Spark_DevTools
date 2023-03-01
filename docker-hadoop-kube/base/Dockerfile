ARG BASE_IMAGE_TAG=10
FROM debian:${BASE_IMAGE_TAG}

#MAINTAINER Ivan Ermilov <ivan.s.ermilov@gmail.com>
#MAINTAINER Giannis Mouchakis <gmouchakis@iit.demokritos.gr>

#RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
#      ca-certificates \
#      net-tools \
#      curl \
#      wget \
#      gnupg \
#      libsnappy-dev \
#    && rm -rf /var/lib/apt/lists/*

# installing common dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive  apt-get install -y --no-install-recommends \
    coreutils ca-certificates build-essential net-tools \
    curl wget gnupg nano\
    gcc g++ gdb lcov pkg-config libbz2-dev \
    libffi-dev libgdbm-dev libgdbm-compat-dev liblzma-dev \
    libncurses-dev libreadline6-dev libsqlite3-dev libssl-dev \
    lzma lzma-dev tk-dev uuid-dev zlib1g-dev libsnappy-dev  \
    && rm -rf /var/lib/apt/lists/*

# installing java
RUN wget https://download.java.net/openjdk/jdk11/ri/openjdk-11+28_linux-x64_bin.tar.gz

RUN tar -xvf openjdk-11+28_linux-x64_bin.tar.gz

RUN mv jdk-11 /usr/local/lib/jdk-11

RUN ln -s /usr/local/lib/jdk-11/bin/java /usr/local/bin/java
      
ENV JAVA_HOME=/usr/local/lib/jdk-11/

# installing python
ARG PYTHON_VERSION=3.9.16

RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
    && tar -xzvf Python-${PYTHON_VERSION}.tgz

RUN cd Python-${PYTHON_VERSION} && ./configure --prefix=/usr/local && make && make install

# installing hadoop
RUN curl -O https://dist.apache.org/repos/dist/release/hadoop/common/KEYS

RUN gpg --import KEYS

ENV HADOOP_VERSION 3.3.4
ENV HADOOP_URL https://www.apache.org/dist/hadoop/common/hadoop-$HADOOP_VERSION/hadoop-$HADOOP_VERSION.tar.gz

RUN set -x \
    && curl -fSL "$HADOOP_URL" -o /tmp/hadoop.tar.gz \
    && curl -fSL "$HADOOP_URL.asc" -o /tmp/hadoop.tar.gz.asc \
    && gpg --verify /tmp/hadoop.tar.gz.asc \
    && tar -xvf /tmp/hadoop.tar.gz -C /opt/ \
    && rm /tmp/hadoop.tar.gz*

RUN ln -s /opt/hadoop-$HADOOP_VERSION/etc/hadoop /etc/hadoop

RUN mkdir /opt/hadoop-$HADOOP_VERSION/logs

RUN mkdir /hadoop-data

ENV HADOOP_HOME=/opt/hadoop-$HADOOP_VERSION
ENV HADOOP_CONF_DIR=/etc/hadoop
ENV MULTIHOMED_NETWORK=1
ENV USER=root
ENV PATH $HADOOP_HOME/bin/:$PATH

ADD entrypoint.sh /entrypoint.sh

RUN chmod a+x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
