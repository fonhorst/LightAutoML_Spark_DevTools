import asyncio
import datetime
import json
import logging
import logging.config
import os
import random
import sys
import typing as t
import uuid
from copy import deepcopy
from dataclasses import dataclass, fields, field
from pprint import pprint

import click
import mlflow
import yaml


def get_logging_config(logfile: str):
    return {
        'version': 1,
        'disable_existing_loggers': True,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stderr',
            },
            'logfile': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.FileHandler',
                'filename': f'{logfile}',
            }
        },
        'loggers': {
            'root': {
                'handlers': ['default'],
                'level': 'DEBUG',
                'propagate': False
            },
            'REPEATER': {
                'handlers': ['default', 'logfile'],
                'level': 'DEBUG',
                'propagate': False
            }
        }
    }


@dataclass(frozen=True)
class RepetitionRun:
    run_uid: uuid.UUID
    cmd: str
    path_to_run_params: str
    spark_submit_exec_path: str
    experiment_script_path: str
    stdout_logfile: str
    workdir: str
    env_params: dict = field(default_factory=dict)

    @property
    def args(self) -> t.List[str]:
        return [self.spark_submit_exec_path, self.experiment_script_path, self.path_to_run_params]

    @property
    def env(self) -> dict:
        return self.env_params


@dataclass
class Parameters:
    dataset_path: t.Optional[str] = None
    run_parameters: t.Optional[dict] = None
    env_parameters: t.Optional[dict] = None

    @staticmethod
    def _convert_str_float_to_float(element: any) -> t.Optional[t.Union[t.Any, float]]:
        if element is None:
            return
        try:
            return float(element)
        except ValueError:
            return element

    @classmethod
    def from_dict(cls, item: dict):
        return cls(**{k: v for k, v in item.items() if k in [field.name for field in fields(cls)]})

    def __post_init__(self):
        self.run_parameters = {k: self._convert_str_float_to_float(v) for k, v in self.run_parameters.items()}
        self.env_parameters = {k: self._convert_str_float_to_float(v) for k, v in self.env_parameters.items()}

    def __hash__(self):
        immutable_dict = [
            (k, self._convert_str_float_to_float(v)) for k, v in
            list(self.run_parameters.items()) + list(self.env_parameters.items())
        ]
        other_params = list(
            {
                k: self._convert_str_float_to_float(v) for k, v in self.__dict__.items()
                if k not in ['run_parameters', 'env_parameters']
            }.items()
        )
        return hash(tuple(sorted(immutable_dict + other_params)))


@dataclass
class ParametersToLog(Parameters):
    mlflow_uri_path: t.Optional[str] = None
    run_tag: t.Optional[str] = 'other'


logger = logging.getLogger("REPEATER")


class Repeater:
    def __init__(
            self,
            configuration: dict,
            stdout_log_dir: str,
            mlflow_tracking_uri: t.Optional[str] = None,
            run_tag: t.Optional[str] = None,
            check_existing_experiments: bool = True
    ):
        current_dt = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.configuration = configuration
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.run_tag = run_tag if run_tag is not None else str(uuid.uuid4())
        self.stdout_log_dir = os.path.join(stdout_log_dir, f"runs_{current_dt}")
        self.check_existing_experiments = check_existing_experiments
        logger.info(f"Running with RUN TAG: {self.run_tag}")

    @staticmethod
    def _check_for_exceptions(done: t.Set[asyncio.Future]) -> None:
        for d in done:
            try:
                d.result()
            except Exception as ex:
                logger.exception(f"Found error in coroutines of processes: {ex}")

    def _save_params(self, path_to_save_params: str, params: Parameters) -> t.Tuple[str, uuid.UUID]:
        assert self.mlflow_tracking_uri is not None, "Mlflow tracking uri should be set"
        run_uuid = uuid.uuid4()
        os.makedirs(path_to_save_params, exist_ok=True)
        updated_parameters = ParametersToLog.from_dict(
            {
                **params.__dict__,
                **{'mlflow_uri_path': self.mlflow_tracking_uri, 'run_tag': self.run_tag, }
            }
        )
        with open(os.path.join(path_to_save_params, f'run_params_{run_uuid}.json'), 'w') as f:
            json.dump(updated_parameters.__dict__, f)
        return os.path.join(path_to_save_params, f'run_params_{run_uuid}.json'), run_uuid

    @staticmethod
    def _collect_existing_runs(runs: t.List[mlflow.entities.Run]) -> t.Set[Parameters]:
        existing_runs = set()
        for run in runs:
            run_params = run.to_dictionary()['data']['params']
            params_run = {
                k[len('run_parameters.'):]: v for k, v in run_params.items() if k.startswith('run_parameters.')
            }
            params_env = {
                k[len('env_parameters.'):]: v for k, v in run_params.items() if k.startswith('env_parameters.')
            }
            other_params = {
                k: v for k, v in run_params.items()
                if not k.startswith('run_parameters.') and not k.startswith('env_parameters.')
            }
            existing_runs.add(
                Parameters.from_dict({**other_params, 'run_parameters': params_run, 'env_parameters': params_env})
            )
        return existing_runs

    def _get_existing_runs(self, mlflow_experiments_ids: t.List[str]) -> t.Set[Parameters]:
        if self.mlflow_tracking_uri is None or not self.check_existing_experiments:
            return set()

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # all_experiments = [
        #     exp.experiment_id for exp in mlflow.search_experiments(filter_string='attribute.name != "Default"')
        # ]
        all_experiments = mlflow_experiments_ids

        runs = mlflow.search_runs(
            experiment_ids=all_experiments,
            filter_string='attributes.status != "FAILED"',
            output_format='list'
        )
        existing_runs = self._collect_existing_runs(runs)
        return existing_runs

    def _prepare_env_params(self, run_config: dict) -> dict:
        env_params = deepcopy(run_config['env_parameters'])
        env_params['EXPERIMENT'] = run_config['mlflow_experiment_id']
        env_params['LOG_TO_MLFLOW'] = '1'
        env_params['MLFLOW_TRACKING_URI'] = self.mlflow_tracking_uri
        return env_params

    def _prepare_configurations(self) -> t.Generator[RepetitionRun, None, None]:
        self._check_for_mlflow()

        logger.info("Finding existing runs for an experiment. Preparing the list of configurations for the run")
        path_to_save_params: str = self.configuration['path_to_save_params']
        configurations: t.List[dict] = self.configuration.get('configuration', None)

        mlflow_exp_ids = list({
            run_config['mlflow_experiment_id']
            for run_config in configurations
            if 'mlflow_experiment_id' in run_config
        })

        existing_runs = self._get_existing_runs(mlflow_exp_ids)
        for run_config in configurations:
            env_params = self._prepare_env_params(run_config)

            run_params = run_config.get('run_parameters', dict())
            current_params = Parameters(
                dataset_path=env_params['DATASET'],
                env_parameters=env_params,
                run_parameters=run_params,
            )
            if current_params in existing_runs:
                logger.info(f"Found existing configuration. {current_params}. Skipping.")
            else:
                path_to_run_params, run_uid = self._save_params(
                    path_to_save_params=path_to_save_params,
                    params=current_params,
                )
                yield RepetitionRun(
                    run_uid=run_uid,
                    cmd=run_config['cmd'],
                    spark_submit_exec_path=run_config['spark_submit_exec_path'],
                    experiment_script_path=run_config['experiment_script_path'],
                    path_to_run_params=path_to_run_params,
                    stdout_logfile=os.path.join(self.stdout_log_dir, f'run-{run_uid}.log'),
                    workdir=run_config['workdir'],
                    env_params=env_params,
                )

    async def _execute_run(self, rep_run: RepetitionRun) -> None:
        logger.info(f"Starting process with uid {rep_run.run_uid}, cmd {rep_run.cmd} and args {rep_run.args}")
        with open(rep_run.stdout_logfile, "w") as f:
            proc = await asyncio.create_subprocess_exec(
                rep_run.cmd,
                *rep_run.args,
                stdout=f,
                stderr=asyncio.subprocess.STDOUT,
                cwd=rep_run.workdir,
                env=rep_run.env,
            )
            ret_code = await proc.wait()

        if ret_code != 0:
            msg = f"Return code {ret_code} != 0 for run with uid {rep_run.run_uid} " \
                  f"with cmd '{rep_run.cmd}' and args '{rep_run.args}'"
            logger.error(msg)
            raise Exception(msg)
        else:
            logger.info(f"Successful run (uid {rep_run.run_uid}) with cmd '{rep_run.cmd}' and args '{rep_run.args}'")

    async def run_repetitions(self, max_parallel_processes: t.Optional[int]):
        logger.info(f"Starting the run with experiment tag {self.run_tag}")
        self._prepare_folders()
        configurations = list(self._prepare_configurations())
        configurations = random.sample(configurations, len(configurations))
        if not len(configurations):
            logger.warning(f"No configurations to calculate. Interrupting execution.")
            return
        processes = [self._execute_run(rep_run) for rep_run in configurations]
        logger.info(f"Initial number of configurations to calculate: {len(configurations)}")
        if max_parallel_processes:
            logger.info(f"Max count of parallel processes are restricted to {max_parallel_processes}")
            total_done_count = 0
            run_slots: t.List[asyncio.Future] = [asyncio.create_task(p) for p in processes[:max_parallel_processes]]
            processes = processes[max_parallel_processes:] if len(processes) > max_parallel_processes else []
            while len(run_slots) > 0:
                done, pending = await asyncio.wait(run_slots, return_when=asyncio.FIRST_COMPLETED)
                self._check_for_exceptions(done)
                total_done_count += len(done)
                free_slots = max_parallel_processes - len(pending)
                run_slots = list(pending) + processes[:free_slots]
                processes = processes[free_slots:] if len(processes) > free_slots else []
                logger.info(f"{total_done_count} configurations have been calculated. "
                            f"{len(configurations) - total_done_count} are left.")
        else:
            logger.info(f"No restrictions on number of parallel processes. "
                        f"Starting all {len(configurations)} configurations.")
            await asyncio.wait(processes)

    def _check_for_mlflow(self):
        assert self.mlflow_tracking_uri is not None

    def _prepare_folders(self):
        os.makedirs(self.stdout_log_dir, exist_ok=True)


class DryRunRepeater(Repeater):
    def _get_existing_runs(self, mlflow_experiments_ids: t.List[str]) -> t.Set[Parameters]:
        return set()

    async def _execute_run(self, rep_run: RepetitionRun) -> None:
        pprint(rep_run)
        await super()._execute_run(rep_run)


@click.command(context_settings=dict(allow_extra_args=True))
@click.option('--config', 'python_config', required=True, help='a path to the config file', type=str)
@click.option('--runs-log-dir',
              default="/var/log", help='a path to the directory where logs of all runs will be stored', type=str)
@click.option('--parallel', default=None,
              required=False, help='a max number of parallel processes running at the same moment', type=int)
@click.option('--log-file', default="/var/log/repeater.log",
              help='a log file to write logs of the algorithm execution to')
@click.option('--tag', required=True, type=str, help='Experiment name for MlFlow experiments')
@click.option('--mlflow-tracking-uri', required=True, help='MlFlow tracking URI')
@click.option('--dry-run', is_flag=True, default=False,  help="Dry Run mode")
def main(
        python_config: str,
        mlflow_tracking_uri: t.Optional[str],
        runs_log_dir: str,
        parallel: t.Optional[int],
        log_file: str,
        tag: t.Optional[str],
        dry_run: bool
):
    logging.config.dictConfig(get_logging_config(log_file))
    logger.info(f"Starting repeater with arguments: {sys.argv}")

    globals_dict = dict()
    with open(python_config, "r") as f:
        exec(f.read(), globals_dict)
        configuration = globals_dict['configurations']

    if dry_run:
        repeater = DryRunRepeater(
            configuration=configuration,
            mlflow_tracking_uri=mlflow_tracking_uri,
            run_tag=tag,
            stdout_log_dir=runs_log_dir,
        )
    else:
        repeater = Repeater(
            configuration=configuration,
            mlflow_tracking_uri=mlflow_tracking_uri,
            run_tag=tag,
            stdout_log_dir=runs_log_dir,
        )
    asyncio.run(repeater.run_repetitions(max_parallel_processes=parallel))
    logger.info("Repeater has finished.")


if __name__ == "__main__":
    main()
