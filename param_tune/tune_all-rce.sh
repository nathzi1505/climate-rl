BASE_DIR="/gws/ssde/j25a/ai4er/users/pn341/climate-rl"

# rce-v0

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "rce-v0-optim-L" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "rce-v0-optim-L" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "rce-v0-optim-L" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "rce-v0-optim-L" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "rce-v0-optim-L" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "rce-v0-optim-L" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "rce-v0-optim-L" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "rce-v0-optim-L" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "rce-v0-optim-L" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sleep 600

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "rce-v0-homo-64L" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "rce-v0-homo-64L" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "rce-v0-homo-64L" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "rce-v0-homo-64L" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "rce-v0-homo-64L" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "rce-v0-homo-64L" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "rce-v0-homo-64L" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "rce-v0-homo-64L" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "rce-v0-homo-64L" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sleep 600

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "rce-v0-optim-L-10k" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "rce-v0-optim-L-10k" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "rce-v0-optim-L-10k" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "rce-v0-optim-L-10k" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "rce-v0-optim-L-10k" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "rce-v0-optim-L-10k" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "rce-v0-optim-L-10k" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "rce-v0-optim-L-10k" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "rce-v0-optim-L-10k" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 10000 --num_steps 500 && sleep 60

# sleep 600

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "rce-v0-homo-64L-10k" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "rce-v0-homo-64L-10k" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "rce-v0-homo-64L-10k" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "rce-v0-homo-64L-10k" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "rce-v0-homo-64L-10k" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "rce-v0-homo-64L-10k" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "rce-v0-homo-64L-10k" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "rce-v0-homo-64L-10k" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "rce-v0-homo-64L-10k" --env_id "RadiativeConvectiveModel-v0" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sleep 600

# rce17-v0

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "rce17-v0-optim-L" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "rce17-v0-optim-L" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "rce17-v0-optim-L" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "rce17-v0-optim-L" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "rce17-v0-optim-L" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "rce17-v0-optim-L" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "rce17-v0-optim-L" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "rce17-v0-optim-L" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "rce17-v0-optim-L" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sleep 600

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "rce17-v0-homo-64L" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "rce17-v0-homo-64L" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "rce17-v0-homo-64L" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "rce17-v0-homo-64L" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "rce17-v0-homo-64L" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "rce17-v0-homo-64L" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "rce17-v0-homo-64L" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "rce17-v0-homo-64L" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "rce17-v0-homo-64L" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sleep 600

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "rce17-v0-optim-L-10k" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "rce17-v0-optim-L-10k" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "rce17-v0-optim-L-10k" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "rce17-v0-optim-L-10k" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "rce17-v0-optim-L-10k" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "rce17-v0-optim-L-10k" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "rce17-v0-optim-L-10k" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "rce17-v0-optim-L-10k" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "rce17-v0-optim-L-10k" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 10000 --num_steps 500 && sleep 60

# sleep 600

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "rce17-v0-homo-64L-10k" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "rce17-v0-homo-64L-10k" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "rce17-v0-homo-64L-10k" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "rce17-v0-homo-64L-10k" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "rce17-v0-homo-64L-10k" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "rce17-v0-homo-64L-10k" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "rce17-v0-homo-64L-10k" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "rce17-v0-homo-64L-10k" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "rce17-v0-homo-64L-10k" --env_id "RadiativeConvectiveModel17-v0" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sleep 600

# rce17-v1

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "rce17-v1-optim-L" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "rce17-v1-optim-L" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "rce17-v1-optim-L" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "rce17-v1-optim-L" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "rce17-v1-optim-L" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "rce17-v1-optim-L" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "rce17-v1-optim-L" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "rce17-v1-optim-L" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "rce17-v1-optim-L" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sleep 600

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "rce17-v1-homo-64L" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "rce17-v1-homo-64L" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "rce17-v1-homo-64L" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "rce17-v1-homo-64L" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "rce17-v1-homo-64L" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "rce17-v1-homo-64L" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "rce17-v1-homo-64L" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "rce17-v1-homo-64L" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "rce17-v1-homo-64L" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sleep 600

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "rce17-v1-optim-L-10k" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "rce17-v1-optim-L-10k" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "rce17-v1-optim-L-10k" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "rce17-v1-optim-L-10k" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "rce17-v1-optim-L-10k" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "rce17-v1-optim-L-10k" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "rce17-v1-optim-L-10k" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "rce17-v1-optim-L-10k" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "rce17-v1-optim-L-10k" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sleep 600

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "rce17-v1-homo-64L-10k" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "rce17-v1-homo-64L-10k" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "rce17-v1-homo-64L-10k" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "rce17-v1-homo-64L-10k" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "rce17-v1-homo-64L-10k" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "rce17-v1-homo-64L-10k" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "rce17-v1-homo-64L-10k" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "rce17-v1-homo-64L-10k" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "rce17-v1-homo-64L-10k" --env_id "RadiativeConvectiveModel17-v1" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60

# rce17-v2

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "rce17-v2-optim-L" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "rce17-v2-optim-L" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "rce17-v2-optim-L" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "rce17-v2-optim-L" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "rce17-v2-optim-L" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "rce17-v2-optim-L" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "rce17-v2-optim-L" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 5000 --num_steps 500 && sleep 60
sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "rce17-v2-optim-L" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "rce17-v2-optim-L" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 5000 --num_steps 500 && sleep 60
# sleep 600

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "rce17-v2-homo-64L" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "rce17-v2-homo-64L" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "rce17-v2-homo-64L" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "rce17-v2-homo-64L" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "rce17-v2-homo-64L" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "rce17-v2-homo-64L" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "rce17-v2-homo-64L" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "rce17-v2-homo-64L" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "rce17-v2-homo-64L" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 5000 --num_steps 500 --homo64 && sleep 60
# sleep 600

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "rce17-v2-optim-L-10k" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "rce17-v2-optim-L-10k" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "rce17-v2-optim-L-10k" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "rce17-v2-optim-L-10k" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "rce17-v2-optim-L-10k" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "rce17-v2-optim-L-10k" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "rce17-v2-optim-L-10k" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 10000 --num_steps 500 && sleep 60
sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "rce17-v2-optim-L-10k" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "rce17-v2-optim-L-10k" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 10000 --num_steps 500 && sleep 60
# sleep 600

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "rce17-v2-homo-64L-10k" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "rce17-v2-homo-64L-10k" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "rce17-v2-homo-64L-10k" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "rce17-v2-homo-64L-10k" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "rce17-v2-homo-64L-10k" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "rce17-v2-homo-64L-10k" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "rce17-v2-homo-64L-10k" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "rce17-v2-homo-64L-10k" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "rce17-v2-homo-64L-10k" --env_id "RadiativeConvectiveModel17-v2" --opt_timesteps 10000 --num_steps 500 --homo64 && sleep 60
