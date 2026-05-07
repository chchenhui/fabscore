#!/usr/bin/env bash

set -u

JUDGE_TYPE="claude"
TASK_ROOT="agents4sci_acc"
DRY_RUN=0
INCLUDE_COMPLETED=0
STOP_ON_ERROR=0
MAX_TASKS=""
GPU_ID="2"

usage() {
  cat <<'EOF'
Usage: ./run_agent4sci_acc.sh [options]

Run FabScore on agent4sci_acc directories that do not yet have
fabscore_<judge_type>/fs_summary.json.

Options:
  --judge-type TYPE       Judge type to pass to main.py (default: claude)
  --task-root PATH        Root task directory (default: agent4sci_acc)
  --max-tasks N           Run at most N selected tasks
  --dry-run               Print selected tasks without executing anything
  --include-completed     Include tasks that already have fs_summary.json
  --stop-on-error         Stop immediately if any task fails
  -h, --help              Show this help message
EOF
}

resolve_paper_filename() {
  local task_dir="$1"
  local pdf_file
  pdf_file=$(find "$task_dir" -maxdepth 1 -type f -iname '*.pdf' | head -n 1)
  if [[ -n "$pdf_file" ]]; then
    printf '%s\n' "$(basename "$pdf_file")"
    return 0
  fi
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --judge-type)
      JUDGE_TYPE="$2"
      shift 2
      ;;
    --task-root)
      TASK_ROOT="$2"
      shift 2
      ;;
    --max-tasks)
      MAX_TASKS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --include-completed)
      INCLUDE_COMPLETED=1
      shift
      ;;
    --stop-on-error)
      STOP_ON_ERROR=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$TASK_ROOT" ]]; then
  echo "Task root does not exist: $TASK_ROOT" >&2
  exit 1
fi

selected_tasks=()
completed_count=0
remaining_count=0
total_count=0

while IFS= read -r -d '' task_dir; do
  total_count=$((total_count + 1))
  summary_path="$task_dir/fabscore_${JUDGE_TYPE}/fs_summary.json"
  if [[ -f "$summary_path" ]]; then
    completed_count=$((completed_count + 1))
    if [[ "$INCLUDE_COMPLETED" -eq 1 ]]; then
      selected_tasks+=("$task_dir")
    fi
  else
    remaining_count=$((remaining_count + 1))
    selected_tasks+=("$task_dir")
  fi
done < <(find "$TASK_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

if [[ -n "$MAX_TASKS" ]]; then
  trimmed_tasks=()
  count=0
  for task_dir in "${selected_tasks[@]}"; do
    if [[ "$count" -ge "$MAX_TASKS" ]]; then
      break
    fi
    trimmed_tasks+=("$task_dir")
    count=$((count + 1))
  done
  selected_tasks=("${trimmed_tasks[@]}")
fi

echo "Discovered $total_count task(s) under $TASK_ROOT."
echo "Completed for judge '$JUDGE_TYPE': $completed_count"
echo "Remaining for judge '$JUDGE_TYPE': $remaining_count"
echo "Using GPU: $GPU_ID"

if [[ ${#selected_tasks[@]} -eq 0 ]]; then
  echo "No tasks selected. Nothing to do."
  exit 0
fi

echo "Selected tasks:"
for task_dir in "${selected_tasks[@]}"; do
  echo "- $task_dir"
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run only. No commands executed."
  exit 0
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"

failures=()
task_index=0
for task_dir in "${selected_tasks[@]}"; do
  task_index=$((task_index + 1))
  if ! paper_filename="$(resolve_paper_filename "$task_dir")"; then
    failures+=("$task_dir (${task_dir##*/}.pdf not found)")
    echo "Task failed: $task_dir (${task_dir##*/}.pdf not found)" >&2
    if [[ "$STOP_ON_ERROR" -eq 1 ]]; then
      break
    fi
    continue
  fi

  command=(uv run python main.py --task_path "$task_dir" --paper_filename "$paper_filename" --judge_type "$JUDGE_TYPE")

  echo
  echo "[$task_index/${#selected_tasks[@]}] Running: ${command[*]}"
  if ! "${command[@]}"; then
    failures+=("$task_dir")
    echo "Task failed: $task_dir" >&2
    if [[ "$STOP_ON_ERROR" -eq 1 ]]; then
      break
    fi
  fi
done

if [[ ${#failures[@]} -gt 0 ]]; then
  echo
  echo "${#failures[@]} task(s) failed:"
  for fail in "${failures[@]}"; do
    echo "- $fail"
  done
  exit 1
fi

echo "All selected tasks completed successfully."