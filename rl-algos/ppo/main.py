import json
import os
import pickle
import random
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Optional

import climate_envs
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from ppo_agent import Agent
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(BASE_DIR)

from param_tune.utils.no_op_summary_writer import NoOpSummaryWriter

os.environ["WANDB__SERVICE_WAIT"] = "600"
os.environ["MUJOCO_GL"] = "egl"
date = time.strftime("%Y-%m-%d", time.gmtime(time.time()))


@dataclass
class Args:
    exp_name: str = "ppo_torch"
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
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    env_id: str = "SimpleClimateBiasCorrection-v0"
    """the id of the environment"""
    total_timesteps: int = 60000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of sequential game environments"""
    num_steps: int = 200
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    optimise: bool = False
    """whether to modify output for hyperparameter optimisation"""
    write_to_file: str = ""
    """filename to write last episode return"""
    optim_group: str = ""
    """folder name under results to load optimised set of params"""
    opt_timesteps: Optional[int] = None
    """timestep duration for one single optimisation run"""

    actor_layer_size: int = 64
    """layer size for the actor network"""
    critic_layer_size: int = 64
    """layer size for the critic network"""

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


def make_env(env_id, idx, capture_video, run_name, gamma, capture_video_freq):
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
        env = gym.wrappers.FlattenObservation(
            env
        )  # deal with dm_control's dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: np.clip(obs, -10, 10)
        )
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(
            env, lambda reward: np.clip(reward, -10, 10)
        )
        return env

    return thunk


args = tyro.cli(Args)
args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.num_minibatches)
args.num_iterations = args.total_timesteps // args.batch_size
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
            i,
            args.capture_video,
            run_name,
            args.gamma,
            args.capture_video_freq,
        )
        for i in range(args.num_envs)
    ]
)
assert isinstance(
    envs.single_action_space, gym.spaces.Box
), "only continuous action space is supported"

agent = Agent(envs, args.actor_layer_size, args.critic_layer_size).to(device)
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

obs = torch.zeros(
    (args.num_steps, args.num_envs) + envs.single_observation_space.shape
).to(device)
actions = torch.zeros(
    (args.num_steps, args.num_envs) + envs.single_action_space.shape
).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)

# 1. start the game
global_step = 0
start_time = time.time()
next_obs, _ = envs.reset(seed=args.seed)
next_obs = torch.Tensor(next_obs).to(device)
next_done = torch.zeros(args.num_envs).to(device)

for iteration in range(1, args.num_iterations + 1):
    if args.anneal_lr:
        frac = 1.0 - (iteration - 1.0) / args.num_iterations
        lr_now = frac * args.learning_rate
        optimizer.param_groups[0]["lr"] = lr_now

    for step in range(0, args.num_steps):
        global_step += args.num_envs
        obs[step] = next_obs
        dones[step] = next_done

        # 2. retrieve the actions
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        # 3. execute the game and log the data.
        next_obs, reward, terminations, truncations, infos = envs.step(
            action.cpu().numpy()
        )
        next_done = np.logical_or(terminations, truncations)
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
            next_done
        ).to(device)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(
                        f"global_step={global_step}, episodic_return={info['episode']['r']}"
                    )
                    writer.add_scalar(
                        "charts/episodic_return",
                        info["episode"]["r"],
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/episodic_length",
                        info["episode"]["l"],
                        global_step,
                    )
                    break

    # 4. bootstrap value if not done
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        last_gae_lam = 0  # generalized advantage estimate
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                next_non_terminal = 1.0 - next_done
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]
            delta = (
                rewards[t]
                + args.gamma * next_values * next_non_terminal
                - values[t]
            )
            advantages[t] = last_gae_lam = (
                delta
                + args.gamma
                * args.gae_lambda
                * next_non_terminal
                * last_gae_lam
            )
        returns = advantages + values

    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    # 5. training
    b_inds = np.arange(args.batch_size)
    clipfracs = []
    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            _, new_log_prob, entropy, new_value = agent.get_action_and_value(
                b_obs[mb_inds], b_actions[mb_inds]
            )
            log_ratio = new_log_prob - b_logprobs[mb_inds]
            ratio = log_ratio.exp()

            # 5a. calculate the mini-batch kl divergence
            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-log_ratio).mean()
                approx_kl = ((ratio - 1) - log_ratio).mean()
                clipfracs += [
                    ((ratio - 1.0).abs() > args.clip_coef)
                    .float()
                    .mean()
                    .item()
                ]

            # 5b. calculate the mini-batch advantages
            mb_advantages = b_advantages[mb_inds]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

            # 5c. calculating policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(
                ratio, 1 - args.clip_coef, 1 + args.clip_coef
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # 5d. calculating value loss
            new_value = new_value.view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (new_value - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    new_value - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()

            # 5e. calculating entropy loss
            entropy_loss = entropy.mean()
            loss = (
                pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.target_kl is not None and approx_kl > args.target_kl:
            break

    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = (
        np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    )

    writer.add_scalar(
        "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
    )
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar(
        "losses/old_approx_kl", old_approx_kl.item(), global_step
    )
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)
    # print("SPS:", int(global_step / (time.time() - start_time)))
    writer.add_scalar(
        "charts/SPS",
        int(global_step / (time.time() - start_time)),
        global_step,
    )

    if iteration == args.num_iterations:
        if args.write_to_file:
            episodic_return = info["episode"]["r"][0]
            with open(args.write_to_file, "wb") as file:
                pickle.dump(
                    {
                        "iterations": args.num_iterations,
                        "last_episodic_return": episodic_return,
                    },
                    file,
                )

envs.close()
writer.close()
