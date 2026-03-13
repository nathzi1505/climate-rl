BASE_DIR="/gws/ssde/j25a/ai4er/users/pn341/climate-rl"

# scbc-v0

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "scbc-v0-optim-L" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 2000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "scbc-v0-optim-L" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 2000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "scbc-v0-optim-L" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 2000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "scbc-v0-optim-L" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 2000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "scbc-v0-optim-L" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 2000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "scbc-v0-optim-L" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 2000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "scbc-v0-optim-L" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 2000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "scbc-v0-optim-L" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 2000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "scbc-v0-optim-L" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 2000 --num_steps 200 && sleep 60
# sleep 600

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "scbc-v0-homo-64L" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 2000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "scbc-v0-homo-64L" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 2000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "scbc-v0-homo-64L" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 2000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "scbc-v0-homo-64L" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 2000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "scbc-v0-homo-64L" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 2000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "scbc-v0-homo-64L" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 2000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "scbc-v0-homo-64L" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 2000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "scbc-v0-homo-64L" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 2000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "scbc-v0-homo-64L" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 2000 --num_steps 200 --homo64 && sleep 60
# sleep 600

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "scbc-v0-optim-L-60k" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 60000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "scbc-v0-optim-L-60k" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 60000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "scbc-v0-optim-L-60k" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 60000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "scbc-v0-optim-L-60k" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 60000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "scbc-v0-optim-L-60k" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 60000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "scbc-v0-optim-L-60k" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 60000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "scbc-v0-optim-L-60k" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 60000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "scbc-v0-optim-L-60k" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 60000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "scbc-v0-optim-L-60k" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 60000 --num_steps 200 && sleep 60

# sleep 600

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "scbc-v0-homo-64L-60k" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 60000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "scbc-v0-homo-64L-60k" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 60000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "scbc-v0-homo-64L-60k" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 60000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "scbc-v0-homo-64L-60k" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 60000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "scbc-v0-homo-64L-60k" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 60000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "scbc-v0-homo-64L-60k" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 60000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "scbc-v0-homo-64L-60k" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 60000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "scbc-v0-homo-64L-60k" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 60000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "scbc-v0-homo-64L-60k" --env_id "SimpleClimateBiasCorrection-v0" --opt_timesteps 60000 --num_steps 200 --homo64 && sleep 60
# sleep 600

# # scbc-v1

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "scbc-v1-optim-L" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 2000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "scbc-v1-optim-L" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 2000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "scbc-v1-optim-L" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 2000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "scbc-v1-optim-L" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 2000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "scbc-v1-optim-L" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 2000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "scbc-v1-optim-L" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 2000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "scbc-v1-optim-L" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 2000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "scbc-v1-optim-L" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 2000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "scbc-v1-optim-L" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 2000 --num_steps 200 && sleep 60
# sleep 600

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "scbc-v1-homo-64L" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 2000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "scbc-v1-homo-64L" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 2000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "scbc-v1-homo-64L" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 2000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "scbc-v1-homo-64L" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 2000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "scbc-v1-homo-64L" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 2000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "scbc-v1-homo-64L" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 2000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "scbc-v1-homo-64L" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 2000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "scbc-v1-homo-64L" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 2000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "scbc-v1-homo-64L" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 2000 --num_steps 200 --homo64 && sleep 60
# sleep 600

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "scbc-v1-optim-L-60k" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 60000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "scbc-v1-optim-L-60k" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 60000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "scbc-v1-optim-L-60k" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 60000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "scbc-v1-optim-L-60k" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 60000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "scbc-v1-optim-L-60k" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 60000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "scbc-v1-optim-L-60k" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 60000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "scbc-v1-optim-L-60k" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 60000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "scbc-v1-optim-L-60k" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 60000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "scbc-v1-optim-L-60k" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 60000 --num_steps 200 && sleep 60
# sleep 600

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "scbc-v1-homo-64L-60k" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 60000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "scbc-v1-homo-64L-60k" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 60000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "scbc-v1-homo-64L-60k" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 60000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "scbc-v1-homo-64L-60k" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 60000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "scbc-v1-homo-64L-60k" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 60000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "scbc-v1-homo-64L-60k" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 60000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "scbc-v1-homo-64L-60k" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 60000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "scbc-v1-homo-64L-60k" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 60000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "scbc-v1-homo-64L-60k" --env_id "SimpleClimateBiasCorrection-v1" --opt_timesteps 60000 --num_steps 200 --homo64 && sleep 60
# sleep 600

# scbc-v2

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "scbc-v2-optim-L" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 2000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "scbc-v2-optim-L" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 2000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "scbc-v2-optim-L" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 2000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "scbc-v2-optim-L" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 2000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "scbc-v2-optim-L" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 2000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "scbc-v2-optim-L" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 2000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "scbc-v2-optim-L" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 2000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "scbc-v2-optim-L" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 2000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "scbc-v2-optim-L" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 2000 --num_steps 200 && sleep 60
# sleep 600

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "scbc-v2-homo-64L" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 2000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "scbc-v2-homo-64L" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 2000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "scbc-v2-homo-64L" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 2000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "scbc-v2-homo-64L" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 2000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "scbc-v2-homo-64L" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 2000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "scbc-v2-homo-64L" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 2000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "scbc-v2-homo-64L" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 2000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "scbc-v2-homo-64L" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 2000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "scbc-v2-homo-64L" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 2000 --num_steps 200 --homo64 && sleep 60
# sleep 600

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "scbc-v2-optim-L-60k" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 60000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "scbc-v2-optim-L-60k" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 60000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "scbc-v2-optim-L-60k" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 60000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "scbc-v2-optim-L-60k" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 60000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "scbc-v2-optim-L-60k" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 60000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "scbc-v2-optim-L-60k" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 60000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "scbc-v2-optim-L-60k" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 60000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "scbc-v2-optim-L-60k" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 60000 --num_steps 200 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "scbc-v2-optim-L-60k" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 60000 --num_steps 200 && sleep 60
# sleep 600

# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo reinforce --exp_id "scbc-v2-homo-64L-60k" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 60000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo dpg --exp_id "scbc-v2-homo-64L-60k" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 60000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ddpg --exp_id "scbc-v2-homo-64L-60k" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 60000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo td3 --exp_id "scbc-v2-homo-64L-60k" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 60000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo ppo --exp_id "scbc-v2-homo-64L-60k" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 60000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo trpo --exp_id "scbc-v2-homo-64L-60k" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 60000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo sac --exp_id "scbc-v2-homo-64L-60k" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 60000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo tqc --exp_id "scbc-v2-homo-64L-60k" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 60000 --num_steps 200 --homo64 && sleep 60
# sbatch $BASE_DIR/param_tune/tune_slurm.sh --algo avg --exp_id "scbc-v2-homo-64L-60k" --env_id "SimpleClimateBiasCorrection-v2" --opt_timesteps 60000 --num_steps 200 --homo64 && sleep 60
