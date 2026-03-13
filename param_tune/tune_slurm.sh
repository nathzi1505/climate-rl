#!/bin/sh

#SBATCH --job-name=pn341_ray_slurm_optimise
#SBATCH --output=/gws/ssde/j25a/ai4er/users/pn341/climate-rl/slurm/ray_slurm_%j.out
#SBATCH --error=/gws/ssde/j25a/ai4er/users/pn341/climate-rl/slurm/ray_slurm_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --account=orchid
#SBATCH --partition=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:2

## SBATCH --account=ai4er
## SBATCH --partition=standard
## SBATCH --qos=high
## SBATCH --nodelist=host[1201-1272]

## SBATCH --account=orchid
## SBATCH --partition=orchid
## SBATCH --qos=orchid
## SBATCH --gres=gpu:2

source ~/miniconda3/etc/profile.d/conda.sh && conda activate venv
BASE_DIR=/gws/ssde/j25a/ai4er/users/pn341/climate-rl
set -x

# 1a. Function to display usage
usage() {
    echo "Usage: sbatch tune_slurm.sh --algo <algo> --exp_id <exp_id> --env_id <env_id> --opt_timesteps <steps> --num_steps <steps> [--homo64]"
    exit 1
}

# 1b. Check if no arguments were passed
if [ "$#" -eq 0 ]; then
    usage
fi

# 1c. Parse command-line arguments
HOMO64=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --algo)
            ALGO="$2"
            shift 2
            ;;
        --exp_id)
            EXP_ID="$2"
            shift 2
            ;;
        --env_id)
            ENV_ID="$2"
            shift 2
            ;;
        --opt_timesteps)
            OPT_TIMESTEPS="$2"
            shift 2
            ;;
        --num_steps)
            NUM_STEPS="$2"
            shift 2
            ;;
        --homo64)
            HOMO64=true
            shift 1
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage
            ;;
    esac
done

# 1d: Print parsed values (for debugging)
echo "algo: $ALGO"
echo "exp_id: $EXP_ID"
echo "env_id: $ENV_ID"
echo "opt_timesteps: $OPT_TIMESTEPS"
echo "num_steps: $NUM_STEPS"
echo "homo64: $HOMO64"

# 1e. Check if all flags are set
if [ -z "$ALGO" ] || [ -z "$EXP_ID" ] || [ -z "$ENV_ID" ] || [ -z "$OPT_TIMESTEPS" ] || [ -z "$NUM_STEPS" ]; then
    echo "Error: All flags are required."
    usage
fi

# __doc_head_address_start__

# checking the conda environment
echo "PYTHON: $(which python)"

# getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}
    else
        head_node_ip=${ADDR[0]}
    fi
    echo "IPv6 address detected. We split the IPv4 address as $head_node_ip"
fi

# __doc_head_address_end__

hrand=$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')
port=$(shuf -i 6381-12580 -n 1)
port=$((port + (hrand % 1000)))

k=$(shuf -i 30-55 -n 1)
min_port=$((k * 1000))
max_port=$((min_port + 99))

ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --min-worker-port=$min_port --max-worker-port=$max_port \
    --num-cpus="${SLURM_CPUS_PER_TASK}" --include-dashboard=False --num-gpus=2 --block & # --num-gpus=2

# optional, though may be useful in certain versions of Ray < 1.0.
sleep 30

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address="$ip_head" \
        --min-worker-port=$min_port --max-worker-port=$max_port \
        --num-cpus="${SLURM_CPUS_PER_TASK}" --num-gpus=2 --block & # --num-gpus=2
    sleep 30
done

if [ "$HOMO64" = true ]; then
    python -u $BASE_DIR/param_tune/tune.py \
        --algo $ALGO \
        --exp_id $EXP_ID \
        --env_id $ENV_ID \
        --opt_timesteps $OPT_TIMESTEPS \
        --num_steps $NUM_STEPS \
        --actor_layer_size 64 \
        --critic_layer_size 64
else
    python -u $BASE_DIR/param_tune/tune.py \
        --algo $ALGO \
        --exp_id $EXP_ID \
        --env_id $ENV_ID \
        --opt_timesteps $OPT_TIMESTEPS \
        --num_steps $NUM_STEPS
fi
