#!/bin/bash

# set -e → exit immediately if any command fails.
# set -u → exit if you try to use an undefined variable.
# set -o pipefail → exit if any command in a pipeline fails.
# This makes the script safer.
set -euo pipefail


# Checks that the user passed at least one argument (the YAML config file).
# Otherwise, prints usage and exits.
if [ $# -lt 1 ]; then
  echo "Usage: $0 <llm_config.yaml> <prompt_config.yaml>"
  exit 1
fi

# Always create mlruns directory
mkdir -p mlruns

# If Slurm is available, also ensure slurmerr and slurmout exist
if command -v sbatch &>/dev/null; then
  mkdir -p slurmerr slurmout
fi

LLM_CONFIG=$1
PROMPT_CONFIG=$2

# Takes config file
# Small inline Python helper to extract YAML values
get_yaml() {
python3 <<EOF
import sys, yaml
config_file = "$1"
with open(config_file) as f:
    data = yaml.safe_load(f)
value = ${2}
if value is None:
    sys.exit(1)
print(value)
EOF
}

# Extract top-level values
LLM_NAME=$(get_yaml "$LLM_CONFIG" "data['name']")
PARTITION=$(get_yaml "$LLM_CONFIG" "data['slurm_params']['partition']")
TIME=$(get_yaml "$LLM_CONFIG" "data['slurm_params']['time']")
GRES=$(get_yaml "$LLM_CONFIG" "data['slurm_params']['gres']")
MEM=$(get_yaml "$LLM_CONFIG" "data['slurm_params']['mem']")
SHOT_TYPE=$(get_yaml "$PROMPT_CONFIG" "data['shot_type']")

JOB_NAME="${LLM_NAME}_${SHOT_TYPE}"

# Create sbatch script
SBATCH_SCRIPT=$(mktemp)

cat <<EOF > "$SBATCH_SCRIPT"
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME}
#SBATCH --gres=${GRES}
#SBATCH --mem=${MEM}
#SBATCH --output=./slurmout/${JOB_NAME}.out
#SBATCH --error=./slurmerr/${JOB_NAME}.err

source ../cs_hs_misinfo_env/bin/activate
python predict_checkworthy.py \\
    llm=${LLM_NAME} \\
    prompt=${SHOT_TYPE} \\
    input.run_name=${JOB_NAME}
EOF

# If Slurm is not available, just print the job script
if ! command -v sbatch &>/dev/null; then
  echo "Script running locally (no batched)"
  bash "$SBATCH_SCRIPT"
else
  sbatch "$SBATCH_SCRIPT"
  echo "Submitted job with config $CONFIG"
fi
