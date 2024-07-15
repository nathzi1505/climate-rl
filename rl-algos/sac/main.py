import json
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass
from typing import Optional

import climate_envs
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
from sac_actor import Actor
from sac_critic import Critic
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

from param_tune.utils.no_op_summary_writer import NoOpSummaryWriter

os.environ["WANDB__SERVICE_WAIT"] = "600"
os.environ["MUJOCO_GL"] = "egl"
date = time.strftime("%Y-%m-%d", time.gmtime(time.time()))


@dataclass
class Args:
    exp_name: str = "sac_torch"
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = "personal-p3jitnath"
    """the entity (team) of wandb's project"""
    wandb_group: str = date
    """the group name under wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    capture_video_freq: int = 100
    """episode frequency at which to capture video"""

    env_id: str = "SimpleClimateBiasCorrection-v0"
    """the environment id of the task"""
    total_timesteps: int = 60000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 1000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = (
        1  # 2 if following Denis Yarats' implementation
    )
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """entropy regularization coefficient"""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    sde_sample_freq: int = int(1e3)  # 1 to disable sde
    """freq of sampling new actor noise"""

    optimise: bool = False
    """whether to modify output for hyperparameter optimisation"""
    write_to_file: str = ""
    """filename to write last episode return"""
    optim_group: str = ""
    """folder name under results to load optimised set of params"""
    opt_timesteps: Optional[int] = None
    """timestep duration for one single optimisation run"""

    actor_layer_size: int = 256
    """layer size for the actor network"""
    critic_layer_size: int = 256
    """layer size for the critic network"""

    num_steps: int = 200
    """the number of steps to run in each environment per policy rollout"""

    def __post_init__(self):
        if self.optimise:
            self.track = False
            self.capture_video = False
            self.total_timesteps = self.opt_timesteps

        if self.optim_group:
            algo = self.exp_name.split("_")[0]
            with open(
                f"{BASE_DIR}/param_tune/results/{self.optim_group}/best_results.json",
                "r",
            ) as file:
                opt_params = {
                    k: v
                    for k, v in json.load(file)[algo].items()
                    if k not in {"algo", "episodic_return", "date"}
                }
                for key, value in opt_params.items():
                    if key == "actor_critic_layer_size":
                        setattr(self, "actor_layer_size", value)
                        setattr(self, "critic_layer_size", value)
                    elif hasattr(self, key):
                        setattr(self, key, value)


def make_env(env_id, seed, idx, capture_video, run_name, capture_video_freq):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                episode_trigger=lambda x: (x == 0)
                or (
                    x % capture_video_freq == (capture_video_freq - 1)
                ),  # add 1 to the episode count generated by gym
            )
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


args = tyro.cli(Args)
run_name = f"{args.wandb_group}/{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

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

if args.optimise:
    writer = NoOpSummaryWriter()
else:
    writer = SummaryWriter(f"runs/{run_name}")

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
print(f"device: {device}")
print(f"actor layer size: {args.actor_layer_size}")
print(f"critic layer size: {args.critic_layer_size}")

# 0. env setup
envs = gym.vector.SyncVectorEnv(
    [
        make_env(
            args.env_id,
            args.seed,
            0,
            args.capture_video,
            run_name,
            args.capture_video_freq,
        )
    ]
)
assert isinstance(
    envs.single_action_space, gym.spaces.Box
), "only continuous action space is supported"

max_action = float(envs.single_action_space.high[0])

actor = Actor(envs, args.actor_layer_size).to(device)
qf1 = Critic(envs, args.critic_layer_size).to(device)
qf2 = Critic(envs, args.critic_layer_size).to(device)
qf1_target = Critic(envs, args.critic_layer_size).to(device)
qf2_target = Critic(envs, args.critic_layer_size).to(device)

qf1_target.load_state_dict(qf1.state_dict())
qf2_target.load_state_dict(qf2.state_dict())

q_optimizer = optim.Adam(
    list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr
)
actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

if args.autotune:
    target_entropy = -torch.prod(
        torch.Tensor(envs.single_action_space.shape).to(device)
    ).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    alpha_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
else:
    alpha = args.alpha

envs.single_observation_space.dtype = np.float32
rb = ReplayBuffer(
    args.buffer_size,
    envs.single_observation_space,
    envs.single_action_space,
    device,
    handle_timeout_termination=False,
)

start_time = time.time()

# 1. start the game
obs, _ = envs.reset(seed=args.seed)
for global_step in range(1, args.total_timesteps + 1):
    # 2. retrieve action(s)
    if global_step < args.learning_starts:
        actions = np.array(
            [envs.single_action_space.sample() for _ in range(envs.num_envs)]
        )
    else:
        if (
            args.sde_sample_freq > 0
            and global_step % args.sde_sample_freq == 0
        ):
            actions, _, _ = actor.get_action(
                torch.Tensor(obs).to(device), noise_reset=True
            )
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
        actions = actions.detach().cpu().numpy()

    # 3. execute the game and log data
    next_obs, rewards, terminations, truncations, infos = envs.step(actions)

    if "final_info" in infos:
        for info in infos["final_info"]:
            print(
                f"global_step={global_step}, episodic_return={info['episode']['r']}"
            )
            writer.add_scalar(
                "charts/episodic_return", info["episode"]["r"], global_step
            )
            writer.add_scalar(
                "charts/episodic_length", info["episode"]["l"], global_step
            )
            break

    # 4. save data to replay buffer
    real_next_obs = next_obs.copy()
    for idx, trunc in enumerate(truncations):
        if trunc:
            real_next_obs[idx] = infos["final_observation"][idx]
    rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

    obs = next_obs

    # 5. training
    if global_step > args.learning_starts:
        data = rb.sample(args.batch_size)

        # 5a. calculate the target_q_values to compare with
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = actor.get_action(
                data.next_observations
            )
            qf1_next_target = qf1_target(
                data.next_observations, next_state_actions
            )
            qf2_next_target = qf2_target(
                data.next_observations, next_state_actions
            )
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - alpha * next_state_log_pi
            )
            target_q_values = data.rewards.flatten() + (
                1 - data.dones.flatten()
            ) * args.gamma * (min_qf_next_target).view(-1)

        # 5b. update both the critics
        qf1_a_values = qf1(data.observations, data.actions).view(-1)
        qf2_a_values = qf2(data.observations, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, target_q_values)
        qf2_loss = F.mse_loss(qf2_a_values, target_q_values)
        qf_loss = qf1_loss + qf2_loss
        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()

        # 5c. update the actor network
        if (
            global_step % args.policy_frequency == 0
        ):  # TD3 delayed update support
            for _ in range(args.policy_frequency):
                pi, log_pi, _ = actor.get_action(data.observations)
                qf1_pi = qf1(data.observations, pi)
                qf2_pi = qf2(data.observations, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # 5d. update the entropy alpha optimizer
                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(data.observations)
                    alpha_loss = (
                        -log_alpha.exp() * (log_pi + target_entropy)
                    ).mean()

                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()
                    alpha = log_alpha.exp().item()

        # 5e. soft-update the target networks
        if global_step % args.target_network_frequency == 0:
            for param, target_param in zip(
                qf1.parameters(), qf1_target.parameters()
            ):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data
                )
            for param, target_param in zip(
                qf2.parameters(), qf2_target.parameters()
            ):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data
                )

        if global_step % 100 == 0:
            writer.add_scalar(
                "losses/qf1_values", qf1_a_values.mean().item(), global_step
            )
            writer.add_scalar(
                "losses/qf2_values", qf2_a_values.mean().item(), global_step
            )
            writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
            writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
            writer.add_scalar(
                "losses/qf_loss", qf_loss.item() / 2.0, global_step
            )
            writer.add_scalar(
                "losses/actor_loss", actor_loss.item(), global_step
            )
            writer.add_scalar("losses/alpha", alpha, global_step)
            # print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar(
                "charts/SPS",
                int(global_step / (time.time() - start_time)),
                global_step,
            )
            if args.autotune:
                writer.add_scalar(
                    "losses/alpha_loss", alpha_loss.item(), global_step
                )

    if global_step == args.total_timesteps:
        if args.write_to_file:
            episodic_return = info["episode"]["r"][0]
            with open(args.write_to_file, "wb") as file:
                pickle.dump(
                    {
                        "timesteps": args.total_timesteps,
                        "last_episodic_return": episodic_return,
                    },
                    file,
                )

envs.close()
writer.close()
