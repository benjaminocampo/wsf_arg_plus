#!/bin/bash

# set -e → exit immediately if any command fails.
# set -u → exit if you try to use an undefined variable.
# set -o pipefail → exit if any command in a pipeline fails.
# This makes the script safer.
set -euo pipefail


# Checks that the user passed at least one argument (the YAML config file).
# Otherwise, prints usage and exits.
if [ $# -lt 1 ]; then
  echo "Usage: $0 <config.yaml>"
  exit 1
fi

CONFIG=$1

# Takes config file
# Small inline Python helper to extract YAML values
get_yaml() {
python3 <<EOF
import sys, yaml
with open("$CONFIG") as f:
    data = yaml.safe_load(f)
value = ${1}
if value is None:
    sys.exit(1)
print(value)
EOF
}

# Extract top-level values
JOB_NAME=$(get_yaml "data['name']")
PARTITION=$(get_yaml "data['slurm_params']['partition']")
TIME=$(get_yaml "data['slurm_params']['time']")
GRES=$(get_yaml "data['slurm_params']['gres']")
MEM=$(get_yaml "data['slurm_params']['mem']")

# Create sbatch script
SBATCH_SCRIPT=$(mktemp)

cat <<EOF > "$SBATCH_SCRIPT"
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME}
#SBATCH --gres=${GRES}
#SBATCH --mem=${MEM}
#SBATCH --output=./out/${JOB_NAME}.out

source ../cs_hs_misinfo_env/bin/activate
python predict_checkworthy.py \\
    llm=${JOB_NAME} \\
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
