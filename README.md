<table>
  <tr align="center">
    <!-- UKRI Logo -->
    <td align="center">
      <a href="https://www.ukri.org/">
      <img src="https://raw.githubusercontent.com/ai4er-cdt/earthquake-predictability/main/assets/images/readme/logo_ukri.png" alt="UKRI Logo" width="400" /></a>
    </td>
    <!-- University of Cambridge Logo -->
    <td align="center">
      <a href="https://www.cam.ac.uk/">
      <img src="https://github.com/ai4er-cdt/earthquake-predictability/blob/main/assets/images/readme/logo_cambridge.png?raw=true" alt="University of Cambridge" width="400" /> </a>
    </td>
    <!-- Met Office Logo -->
    <td align="center">
      <a href="https://www.metoffice.gov.uk/">
      <img src="https://www.metoffice.gov.uk/binaries/content/gallery/metofficegovuk/images/about-us/website/mo_master_for_dark_backg_rbg.png" alt="Met Office Logo" width="400" /> </a>
    </td>
  </tr>
</table>


# Towards improving weather and climate models using reinforcement learning
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11960239.svg)](https://doi.org/10.5281/zenodo.11960239) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1RhgvX5JXzvrH3LB_wJvcOkqjZRkoTvDA/view?usp=sharing)

This GitHub repository contains the code, data, and figures for the Master of Research (MRes) report [**Towards improving weather and climate models using reinforcement learning**](https://drive.google.com/drive/folders/1i8S03_6a_B_y1GjFjJV9OxePbENxxeJz?usp=drive_link).

## Overview

This research explores the integration of reinforcement learning (RL) with idealised climate models, marking a good first effort in AI-assisted climate modelling. Utilising 16 experiments across 8 prominent continuous action, model-free RL algorithms (REINFORCE, DDPG, DPG, TD3, PPO, TRPO, SAC, TQC), the study examines two distinct RL environments: Simple Climate Bias Correction and Radiative-Convective Equilibrium. Performed on the JASMIN HPC, the findings demonstrate the effectiveness of off-policy exploration heavy algorithms like DDPG, TD3, and TQC in Simple Climate Bias Correction, and on-policy exploitation centric algorithms like DPG, PPO, and TRPO in Radiative-Convective Equilibrium. Results show significant bias reductions up to 90%, underscoring the potential of RL-based parameterisation in enhancing the accuracy and efficiency of global climate models.

## Project Structure

```
climate-rl/
│
├── climate-envs/           # Gymansium-based climate environments used in the project
├── datasets/               # Dataset files used in simulations
├── misc/                   # Script files for batch-processing runs on JASMIN
├── notebooks/              # Jupyter notebooks for data analysis and results visualization
├── param_tune/             # Code for Ray-powered parameter tuning via multiple parallel batch jobs
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
   ```shell
   git clone https://github.com/nathzi1505/climate-rl.git
   cd climate-rl
   ```

2. Install dependencies:
   - Using Conda (recommended):
     ```shell
     conda env create -f environment.yml
     conda activate climate-rl-mres
     ```

3. Install the climate RL environments:
    ```shell
    cd climate-envs
    pip install -e .
    cd ..
    ```

4. [Optional] Download runs:
    ```bash
    wget https://zenodo.org/records/11960239/files/runs_2024-06-17_15-38.zip
    unzip -qq runs_2024-06-17_15-38.zip -d runs
    rm -rf runs_2024-06-17_15-38.zip
    ```

## Usage

1. To run an RL algorithm (for eg. DDPG) with an environment (for eg. `RadiativeConvectiveModel-v0`) over 15000 timesteps with 500 steps in each episode.
```
python ./rl-algos/ddpg/main.py --env_id "RadiativeConvectiveModel-v0" --total_timesteps 15000 --num-steps 500
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
