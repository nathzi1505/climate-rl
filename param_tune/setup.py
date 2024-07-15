from setuptools import setup, find_packages

setup(
    name="param_tune",
    version="0.1.0",
    packages=find_packages(
        include=['results', 'results.*', 'utils', 'utils.*'],
        exclude=['tmp', 'tmp.*']
        ),
    py_modules=['config', 'tune', 'tune_slurm', "__init__"],
)
