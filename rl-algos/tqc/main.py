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
import torch.optim as optim
import tyro
from sac_actor import Actor
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tqc_quantile_critic import QuantileCritics

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

from param_tune.utils.no_op_summary_writer import NoOpSummaryWriter

os.environ["WANDB__SERVICE_WAIT"] = "600"
os.environ["MUJOCO_GL"] = "egl"
date = time.strftime("%Y-%m-%d", time.gmtime(time.time()))


# PAPER_N_QUANTILES_TO_DROP = {
#     "Hopper": 5,
#     "Swimmer": 2,
#     "HalfCheetah": 0,
#     "Ant": 2,
#     "Walker2d": 2,
#     "Humanoid": 2,
#     "HopperBulletEnv": 5,
#     "SwimmerBulletEnv": 2,
#     "HalfCheetahBulletEnv": 0,
#     "AntBulletEnv": 2,
#     "Walker2dBulletEnv": 2,
#     "HumanoidBulletEnv": 2,
# }


@dataclass
class Args:
    exp_name: str = "tqc_torch"
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
    """the id of the environment"""
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
    handle_timeout_termination: bool = False
    """treat TimeLimit.truncated == True as done == False"""
    n_quantiles: int = 25
    """the number of quantiles used for each Q Network"""
    n_critics: int = 5
    """the number of Q Networks"""
    use_paper_n_quantiles_to_drop: bool = True
    """number of quantiles to drop"""
    n_quantiles_to_drop: int = 0
    """number of quantiles to drop"""
    learning_starts: int = 1000
    """timestep to start learning"""
    actor_adam_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    critic_adam_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    alpha_adam_lr: float = 3e-4
    """the learning rate to tune target entropy"""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = (
        1  # 2 if following Denis Yarats' implementation
    )
    """the frequency of updates for the target nerworks"""

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
    critic_layer_size: int = 512
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


def quantile_huber_loss(quantiles, samples):
    pairwise_delta = (
        samples[:, None, None, :] - quantiles[:, :, :, None]
    )  # batch x nets x quantiles x samples
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(
        abs_pairwise_delta > 1,
        abs_pairwise_delta - 0.5,
        pairwise_delta**2 * 0.5,
    )

    n_quantiles = quantiles.shape[2]
    taus = (
        torch.arange(n_quantiles, device=quantiles.device).float().unsqueeze(0)
        / n_quantiles
        + 1 / 2 / n_quantiles
    )
    elementwise_loss = torch.abs(taus[:, None, :, None] - (pairwise_delta < 0).float()) * huber_loss  # type: ignore
    return elementwise_loss.mean()


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
if args.use_paper_n_quantiles_to_drop:
    print(
        f"Using paper n_quantiles_to_drop: {args.n_quantiles_to_drop} for env: {args.env_id}"
    )

actor = Actor(envs, args.actor_layer_size).to(device)
critics = QuantileCritics(
    envs, args.n_quantiles, args.n_critics, args.critic_layer_size
).to(device)
args.n_top_quantiles_to_drop = args.n_quantiles_to_drop * critics.n_critics
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_adam_lr)
critic_optimizer = torch.optim.Adam(
    critics.parameters(), lr=args.critic_adam_lr
)

target_critics = QuantileCritics(
    envs, args.n_quantiles, args.n_critics, args.critic_layer_size
).to(device)
target_critics.load_state_dict(critics.state_dict())
target_critics.requires_grad_(False)

if args.autotune:
    target_entropy = -torch.prod(
        torch.Tensor(envs.single_action_space.shape).to(device)
    ).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_adam_lr)
else:
    alpha = args.alpha

envs.single_observation_space.dtype = np.float32
rb = ReplayBuffer(
    args.buffer_size,
    envs.single_observation_space,
    envs.single_action_space,
    device,
    handle_timeout_termination=args.handle_timeout_termination,
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
        batch_size = data.rewards.size(0)

        # 5a. calculate the target_q_values to compare with
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = actor.get_action(
                data.next_observations
            )
            next_z = target_critics(data.next_observations, next_state_actions)
            sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
            sorted_z = sorted_z[
                :, : critics.n_total_quantiles - args.n_top_quantiles_to_drop
            ]
            target_z = data.rewards + (1 - data.dones) * args.gamma * (
                sorted_z - alpha * next_state_log_pi
            )

        # 5b. update both the critics
        cur_z = critics(data.observations, data.actions)
        critic_loss = quantile_huber_loss(cur_z, target_z)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # 5c. update the actor network
        if (
            global_step % args.policy_frequency == 0
        ):  # TD3 delayed update support
            for _ in range(args.policy_frequency):
                sampled_actions, log_pi, _ = actor.get_action(
                    data.observations
                )
                actor_loss = (
                    alpha * log_pi
                    - critics(data.observations, sampled_actions)
                    .mean(2)
                    .mean(1, keepdim=True)
                ).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # 5d. update the entropy alpha optimizer
                if args.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = actor.get_action(data.observations)
                    alpha_loss = (
                        -log_alpha * (log_pi + target_entropy).detach().mean()
                    )
                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()
                    alpha = log_alpha.exp().item()

        # 5e. soft-update the target networks
        if global_step % args.target_network_frequency == 0:
            for param, target_param in zip(
                critics.parameters(), target_critics.parameters()
            ):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data
                )

        if global_step % 100 == 0:
            writer.add_scalar("losses/cur_z", cur_z.mean().item(), global_step)
            writer.add_scalar(
                "losses/critic_loss", critic_loss.item(), global_step
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
