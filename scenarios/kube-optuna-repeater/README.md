## Repeater job for Optuna experiments

1. Create PV and PVC by running
```bash
kubectl apply -f experimental-cluster/optuna-pv.yml
kubectl apply -f experimental-cluster/optuna-pvc.yml
```
2. Change configuration file `../experiments/configs/optuna-config.yml` according to the run requirements. 
3. Start run:
```bash
cd .. 
bash bin/experimentctl start-optuna 
```
Starting the run will create 3 ConfigMaps:
- `optuna-script` (from `experiments/scripts/parallel-optuna.py`)
- `optuna-config` (from `experiments/configs/optuna-config.yml`)
- `optuna-spark-submit` (from `experiments/optuna-spark-submit`)

and will start the job, skipping the configurations that have already finished successfully.

