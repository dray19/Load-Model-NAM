set -euo pipefail
IFS=$'\n\t'

usage() {
  echo "Usage: $0 YYYYMMDD"
  echo "Example: $0 20251006"
  exit 2
}

# Validate argument
if [ "${#}" -ne 1 ]; then
  usage
fi

date_arg1="$1"

if ! [[ "$date_arg1" =~ ^[0-9]{8}$ ]]; then
  echo "Error: date must be in YYYYMMDD format (8 digits). Got: '$date_arg1'"
  exit 1
fi

# Use the full YYYYMMDD provided by the user
first_of_month="${date_arg1}06"

# List of model directories to run (edit as needed)
models=(
  "model_5y_lin_bias"
  "model_5y_1y"
  "model_4y_lin_bias"
  "model_4y_1y"
  "model_1y_4m"
  "model_6m_2w_dc_bias"
  "model_6m_dc_bias"
)

# Main loop
for model in "${models[@]}"; do
  if [ ! -d "$model" ]; then
    echo "Warning: directory '$model' not found — skipping."
    continue
  fi

  echo "==> Entering $model"
  pushd "$model" >/dev/null
  if [ -x "./run.sh" ]; then
    ./run.sh "$first_of_month"
  else
    sh ./run.sh "$first_of_month"
  fi
  popd >/dev/null

  # Update forecast config (script assumed to be in repo root)
  if python -c 'import sys' >/dev/null 2>&1; then
    python change_fcst_config.py "Production/config.ini" "$model" "$first_of_month"
  else
    echo "Error: python not found in PATH"
    exit 1
  fi
done

echo "All models processed for ${first_of_month}."