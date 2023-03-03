## Environmental variables used in an experiment
- `PATH_TO_SAVE_RESULTS` - path used to drop results in a CSV format;
- `PARTITION_NUM` - number of partitions, will be set to number of available cpus if not provided;
- `SEED` - random seed used in an experiments;
- `DATASET_PATH` - path to the MovieLens dataset in CSV format


   Requirements:
- `Python 3.8`
- libstdc++.so.6: `GLIBCXX_3.4.26`

## Local experiment launch
0. Clone `pygloo` repo, clone and build RePlay 
   ```bash
   git clone https://github.com/ray-project/pygloo.git
   git clone https://github.com/fonhorst/RePlay.git && cd RePlay && poetry build
   cd ..
   ```
1. Build image and run the container
   ```bash
   cd docker-images && docker build -t local-spark-lightfm . && cd .. 
   export PROJECTROOT=~/LightAutoML_Spark_DevTools/mpi-lightfm/replay-lightfm-distributed
   export PYGLOOROOT=~/pygloo
   export REPLAYROOT=~/RePlay
   
   docker run --rm -it \
   -v $REPLAYROOT/dist:/src/dist \
   -v $PYGLOOROOT/pygloo:/src/pygloo \
   -v $PROJECTROOT/example-spark:/src/example-local-spark \
   -v $PROJECTROOT/ml1m_ratings.csv:/src/ml1m_ratings.csv \
   -p 4040:4040 \
   --entrypoint /bin/bash local-spark-lightfm
   ```
2. In the container, create symbolic link to python
   ```bash
   ln -s /usr/local/bin/python3 /usr/local/bin/python
   ```
3. Install RePlay and its requirements
   ```bash
   python -m pip install src/dist/replay_rec-0.10.0-py3-none-any.whl
   ```
4. Download bazel and add it to path
   ```bash
   wget https://github.com/bazelbuild/bazelisk/releases/download/v1.16.0/bazelisk-linux-amd64
   chmod +x bazelisk-linux-amd64
   mv bazelisk-linux-amd64 /usr/local/bin/bazel
   ```
5. Install `pygloo`
   ```bash
   cd /src/pygloo/ && python setup.py install
   cd dist/ \
   && mv pygloo-0.2.0-py3.8-linux-x86_64.egg pygloo-0.2.0-py3.8-linux-x86_64.egg.zip \
   && unzip pygloo-0.2.0-py3.8-linux-x86_64.egg.zip
   ```
6. Launch local Spark experiment
   ```bash
   PYTHONPATH=$PYTHONPATH:/src/example-local-spark python /src/example-local-spark/run_experiment.py
   ``` 
