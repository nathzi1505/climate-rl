<table>
  <tr align="center">
    <!-- UKRI Logo -->
    <td align="center">
      <a href="https://www.ukri.org/">
      <img src="assets/ukri-logo-coloured.png" alt="UKRI Logo" width="400" /></a>
    </td>
    <!-- University of Cambridge Logo -->
    <td align="center">
      <a href="https://www.cam.ac.uk/">
      <img src="assets/cambridge-logo-coloured.png" alt="University of Cambridge" width="400" /> </a>
    </td>
    <!-- Met Office Logo -->
    <td align="center">
      <a href="https://www.metoffice.gov.uk/">
      <img src="assets/met-office-logo-coloured.png" alt="Met Office Logo" width="400" /> </a>
    </td>
  </tr>
</table>


# RAIN: Reinforcement Algorithms for Improving Numerical Weather and Climate Models
[![arXiv](https://img.shields.io/badge/cs.LG-2408.16118-b31b1b?logo=arXiv&logoColor=red)](https://arxiv.org/abs/2408.16118) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17116349.svg)](https://doi.org/10.5281/zenodo.11960239) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1RhgvX5JXzvrH3LB_wJvcOkqjZRkoTvDA/view?usp=sharing)

This GitHub repository contains the code, data, and figures for the paper [**RAIN: Reinforcement Algorithms for Improving Numerical Weather and Climate Models**](https://arxiv.org/abs/2408.16118). Also includes the SCBC and RCE experiments from the paper [**Learning State-Dependent Weather and Climate Model Parametrisations with Reinforcement Learning**](#).

## Overview

This study investigates how reinforcement learning (RL) can improve parameterisation in idealised climate models, addressing uncertainties from unresolved sub-grid processes. We benchmark eight RL algorithms on two testbeds: temperature bias correction and radiative–convective equilibrium (RCE), finding that exploration-based methods perform better for bias correction, while exploitation-focused methods excel in RCE. The results highlight RL’s potential for enhancing climate model accuracy and efficiency, offering a step towards integrating adaptive parameterisations into global models.

## Project Structure

```
climate-rl/
│
├── assets/                 # README.md assets
├── climate-envs/           # Gymansium-based climate environments used in the project
├── datasets/               # Dataset files used in simulations
├── misc/                   # Script files for batch-processing runs on JASMIN
├── notebooks/              # Jupyter notebooks for data analysis and results visualization
├── param_tune/             # Code for Ray-powered parameter tuning via multiple parallel batch jobs
├── results/                # Results (imgs and tables) for documentation
├── rl-algos/               # cleanRL-styled source code for RL models
├──.editorconfig            # Config file for code editor specific settings
├──.gitignore               # Config file to skip certain files from version control
├──.pre-commit-config.yaml  # Config file for pre-commit tools for maintaining code quality
├── environment.yml         # Conda environment file
└── pyproject.toml          # Config file for python project
```

## Environment Setup

To set up the project environment, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/p3jitnath/climate-rl.git
   cd climate-rl
   ```

2. Install dependencies:
   - Using Conda (recommended):
     ```bash
     conda env create -f environment.yml
     conda activate venv
     ```

3. Replace the `BASE_DIR` location and the conda environment name:
   ```
   find . -type f -exec sed -i "s|/gws/ssde/j25a/ai4er/users/pn341/climate-rl|$(pwd)|g" {} +
   ```

4. Install the climate RL environments:
    ```
    cd climate-envs/ && pip install . && cd ../
    ```

5. [Optional] Download runs:
    ```bash
    wget https://zenodo.org/records/17116349/files/scbc-rce_runs_2025-09-13.zip
    unzip -qq scbc-rce_runs_2025-09-13.zip
    rm -rf scbc-rce_runs_2025-09-13.zip
    ```

## Usage

1. To run an RL algorithm (for eg. DDPG) with an environment (for eg. `RadiativeConvectiveModel-v0`) over 10000 timesteps with 500 steps in each episode.
```
python ./rl-algos/ddpg/main.py --env_id "RadiativeConvectiveModel-v0" --total_timesteps 10000 --num-steps 500
```
> [!NOTE]
> Max. value for `num-steps` can be found [here](/climate-envs/climate_envs/__init__.py).

2. For detailed information regarding each option passed to the algorithm use `-h`.
```
python ./rl-algos/ddpg/main.py -h
```
> [!NOTE]
> Command line examples to run RL algorithms using SLURM can be found in `run` files [here](/misc/).

3. A demo Colab notebook to run `SimpleClimateBiasCorrection-v0` can be found [here](https://drive.google.com/file/d/1RhgvX5JXzvrH3LB_wJvcOkqjZRkoTvDA/view?usp=sharing).


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Funding

P. Nath was supported by the [UKRI Centre for Doctoral Training in Application of Artificial Intelligence to the study of Environmental Risks](https://ai4er-cdt.esc.cam.ac.uk/) [EP/S022961/1].

## Contact

For any queries or further information, please contact [Pritthijit Nath](mailto:pn341@cam.ac.uk).
