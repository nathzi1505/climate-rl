import json

from gymnasium.envs.registration import register

BASE_DIR = "/gws/ssde/j25a/ai4er/users/pn341/climate-rl"

register(
    id="SimpleClimateBiasCorrection-v0",
    entry_point="climate_envs.envs:SimpleClimateBiasCorrectionEnv",
    max_episode_steps=200,
)

register(
    id="SimpleClimateBiasCorrection-v1",
    entry_point="climate_envs.envs.simple_climate_bias_correction_v1:SimpleClimateBiasCorrectionEnv",
    max_episode_steps=200,
)

register(
    id="SimpleClimateBiasCorrection-v2",
    entry_point="climate_envs.envs.simple_climate_bias_correction_v2:SimpleClimateBiasCorrectionEnv",
    max_episode_steps=200,
)

register(
    id="RadiativeConvectiveModel-v0",
    entry_point="climate_envs.envs:RadiativeConvectiveModelEnv",
    max_episode_steps=500,
)

register(
    id="RadiativeConvectiveModel17-v0",
    entry_point="climate_envs.envs.radiative_convective_model_17:RadiativeConvectiveModelEnv",
    max_episode_steps=500,
)

register(
    id="RadiativeConvectiveModel17-v1",
    entry_point="climate_envs.envs.radiative_convective_model_17_v1:RadiativeConvectiveModelEnv",
    max_episode_steps=500,
)
