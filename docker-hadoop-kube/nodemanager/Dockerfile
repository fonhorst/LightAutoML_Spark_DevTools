ARG IMAGE_TAG=slama-yarn
FROM node2.bdcl:5000/yarn-base:${IMAGE_TAG}

HEALTHCHECK CMD curl -f http://localhost:8042/ || exit 1

ADD run.sh /run.sh
RUN chmod a+x /run.sh

EXPOSE 8042

RUN apt update

RUN apt install build-essential libncurses5-dev zlib1g-dev libnss3-dev libgdbm-dev libssl-dev libsqlite3-dev libffi-dev libreadline-dev curl libbz2-dev wget -y



CMD ["/run.sh"]
