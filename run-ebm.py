# run-pytorch-data.py
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
from azureml.core import Dataset
import os

if __name__ == "__main__":
    ws = Workspace.from_config()
    # datastore = ws.get_default_datastore()
    # dataset = Dataset.File.from_files(path=(datastore, 'datasets/cifar10'))
    
    logs_dir = os.path.join(os.curdir, os.path.join("logs", "tb-logs"))
    experiment = Experiment(workspace=ws, name='experiment-ebm')

    config = ScriptRunConfig(
        source_directory='./src',
        script='main.py',
        compute_target='gpu-cluster-ebm',
        arguments=[
            '--sampler', 'cycsgld',
            '--logdir', logs_dir,
            '--datadir', '.data/',
            '--download', True,
            '--step_lr', 20.5,
            '--log_interval', 100,
            '--epoch_num', 100,
            '--dataset', 'cifar10'],
    )

    # use curated pytorch environment
    system_env = Environment.get(workspace=ws,name="EBM-pytorch-1.9-ubuntu18.04-py37-cuda11-gpu",version="4")
    # system_env = Environment.from_pip_requirements( name="EBM-pytorch-1.9-ubuntu18.04-py37-cuda11-gpu", file_path='./requirements.txt')
    # system_env.python.user_managed_dependencies = False
    # system_env.docker.base_image = 'mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04:20210906.v1'
    # system_env.register(workspace=ws)
    config.run_config.environment = system_env
    run = experiment.submit(config)
    aml_url = run.get_portal_url()
    print("Submitted to compute cluster. Click link below")
    print("")
    print(aml_url)