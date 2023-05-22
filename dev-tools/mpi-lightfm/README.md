# MPI LightFM version

Распределенная версия алгоритма LightFM с применением OpenMPI 

---
Distributed algorithm version of LightFM with OpenMPI integration

---

Для запуска алгоритма требуется установка [OpenMPI](https://www.open-mpi.org/) 
и [mpi4py](https://mpi4py.readthedocs.io/en/stable/):

```bash
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.4.tar.bz2
tar -jxf openmpi-4.1.4.tar.bz2

cd openmpi-4.1.4
./configure --prefix=$HOME/opt/openmpi
make all
make install

env MPICC=$HOME/opt/openmpi/bin/mpicc python -m pip install mpi4py
export PATH=$PATH:$HOME/opt/openmpi/bin/
```
