import ast
import glob
import importlib
import json
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import climate_envs
import gymnasium as gym
import numpy as np
import torch
import tyro
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = "/gws/ssde/j25a/ai4er/users/pn341/climate-rl"
sys.path.append(BASE_DIR)

os.environ["WANDB__SERVICE_WAIT"] = "600"
os.environ["MUJOCO_GL"] = "egl"
date = time.strftime("%Y-%m-%d", time.gmtime(time.time()))


@dataclass
class Args:
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = "personal-p3jitnath"
    """the entity (team) of wandb's project"""
    wandb_group: str = date
    """the group name under wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    capture_video_freq: int = 10
    """episode frequency at which to capture video"""

    env_id: str = "SimpleClimateBiasCorrection-v0"
    """the environment id of the environment"""
    algo: str = "ddpg"
    """the RL algorithm to be used"""
    episodes: int = 1
    """total episode count for inference"""
    gamma: float = 0.99
    """the discount factor gamma"""

    optim_group: str = ""
    """folder name under results to load optimised set of params"""

    actor_layer_size: int = 64
    """layer size for the actor network"""
    critic_layer_size: int = 64
    """layer size for the critic network"""

    num_steps: int = 200
    """the number of steps to run in each environment per policy rollout"""
    record_step: int = 60000
    """step count to load the run record"""

    def __post_init__(self):
        self.exp_name = f"inference_{self.algo}_torch"
        self.total_timesteps = self.episodes * self.num_steps
        self.exp_id = self.wandb_group.split("_")[1]

        if self.optim_group:
            if "64L" in self.optim_group:
                self.actor_layer_size = self.critic_layer_size = 64
            else:
                with open(
                    f"{BASE_DIR}/param_tune/results/{self.optim_group}/best_results.json",
                    "r",
                ) as file:
                    opt_params = {
                        k: v
                        for k, v in json.load(file)[self.algo].items()
                        if k not in {"algo", "episodic_return", "date"}
                    }
                    for key, value in opt_params.items():
                        if key == "actor_critic_layer_size":
                            self.actor_layer_size = self.critic_layer_size = (
                                value
                            )


class RecordSteps:
    def __init__(self, steps_folder):
        self.steps_folder = steps_folder
        self._clear()

    def _clear(self):
        self.global_steps = []
        self.obs = []
        self.next_obs = []
        self.actions = []
        self.rewards = []

    def reset(self):
        self._clear()

    def add(self, global_step, obs, next_obs, actions, rewards):
        self.global_steps.append(global_step)
        self.obs.append(obs)
        self.next_obs.append(next_obs)
        self.actions.append(actions)
        self.rewards.append(rewards)

    def save(self, global_step, episodic_return):
        torch.save(
            {
                "global_steps": np.array(self.global_steps).squeeze(),
                "obs": np.array(self.obs).squeeze(),
                "next_obs": np.array(self.next_obs).squeeze(),
                "actions": np.array(self.actions).squeeze(),
                "rewards": np.array(self.rewards).squeeze(),
                "episodic_return": episodic_return,
            },
            f"{self.steps_folder}/step_{global_step}.pth",
        )
        self.reset()


def get_make_env(algo):
    file_path = Path(f"{BASE_DIR}/rl-algos/{algo}/main.py").resolve()
    source = file_path.read_text()

    parsed = ast.parse(source, filename=str(file_path))
    func_defs = [
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name == "make_env"
    ]

    if not func_defs:
        raise ValueError(f"'make_env' not found in {file_path}")

    make_env_code = ast.Module(body=func_defs, type_ignores=[])
    compiled = compile(make_env_code, filename=str(file_path), mode="exec")

    local_namespace = {"gym": gym, "np": np, "BASE_DIR": BASE_DIR}
    exec(compiled, local_namespace)
    return local_namespace["make_env"]


def get_actor(algo):
    module_path = f"rl-algos.{algo}.{algo}_actor"
    actor_module = importlib.import_module(module_path)
    Actor = getattr(actor_module, "Actor")
    return Actor


def get_agent(algo):
    module_path = f"rl-algos.{algo}.{algo}_agent"
    agent_module = importlib.import_module(module_path)
    Agent = getattr(agent_module, "Agent")
    return Agent


args = tyro.cli(Args)
run_name = f"{args.wandb_group}/{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

records_folder = f"{BASE_DIR}/records/{run_name}"
os.makedirs(records_folder, exist_ok=True)
rs = RecordSteps(records_folder)

if args.track:
    import wandb

    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        group=args.wandb_group,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )

writer = SummaryWriter(f"{BASE_DIR}/runs/{run_name}")

writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s"
    % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

device = torch.device(
    "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
)
print(f"device: {device}", flush=True)
print(f"actor layer size: {args.actor_layer_size}", flush=True)

# 0. env setup
make_env = get_make_env(args.algo)
env_args = [
    args.env_id,
    args.seed,
    0,
    args.capture_video,
    run_name,
]
if args.algo in ["ppo", "trpo"]:
    env_args = env_args + [args.gamma, args.capture_video_freq]
else:
    env_args = env_args + [args.capture_video_freq]
envs = gym.vector.SyncVectorEnv([make_env(*env_args)])
assert isinstance(
    envs.single_action_space, gym.spaces.Box
), "only continuous action space is supported"

record_fn = glob.glob(
    f"{BASE_DIR}/records/*{args.exp_id}_*/*_{args.algo}_*{args.seed}_*/*{args.record_step}.pth"
)[0]

if args.algo == "ppo":
    Agent = get_agent(args.algo)
    agent = Agent(envs, args.actor_layer_size, args.critic_layer_size).to(
        device
    )
    record_steps = torch.load(record_fn)
    agent.load_state_dict(record_steps["agent"])
    actor = agent
else:
    Actor = get_actor(args.algo)
    actor = Actor(envs, args.actor_layer_size).to(device)
    record_steps = torch.load(record_fn)
    actor.load_state_dict(record_steps["actor"])

envs.single_observation_space.dtype = np.float32
start_time = time.time()

# 1. start the game
obs, _ = envs.reset(seed=args.seed)
for global_step in range(1, args.total_timesteps + 1):

    # 2. retrieve action(s)
    with torch.no_grad():
        if args.algo in ["dpg", "ddpg", "td3"]:
            actions = actor(torch.Tensor(obs).to(device))
        elif args.algo in ["reinforce", "tqc", "trpo"]:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
        elif args.algo in ["avg"]:
            (
                actions,
                _,
            ) = actor(torch.Tensor(obs).to(device))
        elif args.algo in ["ppo"]:
            actions, _, _, _ = agent.get_action_and_value(
                torch.Tensor(obs).to(device)
            )
        elif args.algo in ["sac"]:
            actions, _, _ = actor.get_action(
                torch.Tensor(obs).to(device), noise_reset=True
            )
        actions = (
            actions.cpu()
            .numpy()
            .clip(envs.single_action_space.low, envs.single_action_space.high)
        )

    # 3. execute the game and log data
    next_obs, rewards, terminations, truncations, infos = envs.step(actions)

    if "final_info" in infos:
        for info in infos["final_info"]:
            print(
                f"seed={args.seed}, global_step={global_step}, episodic_return={info['episode']['r']}",
                flush=True,
            )
            writer.add_scalar(
                "charts/episodic_return", info["episode"]["r"], global_step
            )
            writer.add_scalar(
                "charts/episodic_length", info["episode"]["l"], global_step
            )
            if global_step % args.num_steps == 0:
                rs.save(global_step, info["episode"]["r"])
            break

    real_next_obs = next_obs.copy()
    for idx, trunc in enumerate(truncations):
        if trunc:
            real_next_obs[idx] = infos["final_observation"][idx]
    rs.add(global_step, obs, next_obs, actions, rewards)
    obs = next_obs

envs.close()
writer.close()
