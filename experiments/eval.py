"""
Evaluate experiment variants with three modes:
  --sample   : text completions (default: on)
  --viz      : attention visualization
  --bench    : DCLM CORE benchmark

Each experiment may have been trained on a different codebase version (different
attention implementations, etc.), so we check out the correct commit into a
temporary git worktree and run eval from there.

Examples:
    python experiments/eval.py                          # sample only
    python experiments/eval.py --viz --no-sample        # viz only
    python experiments/eval.py --bench --no-sample      # bench only
    python experiments/eval.py --sample --viz --bench   # all three
"""
import os
import sys
import json
import argparse
import subprocess
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PYTHON = os.path.join(REPO_ROOT, ".venv", "bin", "python")

# ---- Define experiments to evaluate ----
# Each entry: (model_tag, commit, description)
EXPERIMENTS = [
    ("d12-reference",      None,      "baseline softmax (flash attn)"),
    ("d12-reference-v2",   "d910b6a", "softmax via linear_attn codepath"),
    ("d12-linear",         "a9715b0", "relu + x^2 + normalization"),
    ("d12-relu-sq",        "19cdf8f", "relu(x)^2 feature map + row normalization"),
    ("d12-linear-poly-v5", "56253f1", "relu + ax+x^2+bx^4 (learnable a,b)"),
    ("d12-simple-x2",      "7e9eebc", "x^2 + normalization"),
    ("d12-simple-relu",    "2331455", "relu + normalization"),
    ("d12-simple-x4",      "e97b5d5", "x^4 + normalization"),
    ("d12-taylor-relu",    "62a6e70", "relu(1+x+x^2/2+x^3/6)")

]

PROMPT = "The capital of France is"
MAX_TOKENS = 32

# ---------------------------------------------------------------------------
# Worker scripts — these run inside each worktree subprocess
# ---------------------------------------------------------------------------

SAMPLE_WORKER_SCRIPT = r"""
import sys, json, torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

model_tag = sys.argv[1]
prompt = sys.argv[2]
max_tokens = int(sys.argv[3])

device_type = autodetect_device_type()
_, _, _, _, device = compute_init(device_type)
model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=model_tag)
engine = Engine(model, tokenizer)
tokens = tokenizer(prompt, prepend="<|bos|>")
sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=max_tokens, temperature=0)
completion = tokenizer.decode(sample[0][len(tokens):])
print(json.dumps({"completion": completion}))
"""

CORE_WORKER_SCRIPT = r"""
import sys, os, json, csv, time, random, yaml, torch
from nanochat.common import compute_init, autodetect_device_type, get_base_dir, download_file_with_lock
from nanochat.checkpoint_manager import load_model
from nanochat.core_eval import evaluate_task

model_tag = sys.argv[1]
max_per_task = int(sys.argv[2])

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

def place_eval_bundle(file_path):
    import shutil, zipfile, tempfile
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        shutil.move(os.path.join(tmpdir, "eval_bundle"), eval_bundle_dir)

device_type = autodetect_device_type()
_, _, _, _, device = compute_init(device_type)
model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=model_tag)

base_dir = get_base_dir()
eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
if not os.path.exists(eval_bundle_dir):
    download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)

config_path = os.path.join(eval_bundle_dir, "core.yaml")
data_base_path = os.path.join(eval_bundle_dir, "eval_data")
eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
tasks = config['icl_tasks']

random_baselines = {}
with open(eval_meta_data, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        random_baselines[row['Eval Task']] = float(row['Random baseline'])

results = {}
centered_results = {}
for task in tasks:
    label = task['label']
    task_meta = {
        'task_type': task['icl_task_type'],
        'dataset_uri': task['dataset_uri'],
        'num_fewshot': task['num_fewshot'][0],
        'continuation_delimiter': task.get('continuation_delimiter', ' ')
    }
    data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]

    shuffle_rng = random.Random(1337)
    shuffle_rng.shuffle(data)
    if max_per_task > 0:
        data = data[:max_per_task]

    accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
    results[label] = accuracy
    rb = random_baselines[label]
    centered_results[label] = (accuracy - 0.01 * rb) / (1.0 - 0.01 * rb)
    print(f"  {label}: accuracy={accuracy:.4f} centered={centered_results[label]:.4f}", file=sys.stderr)

core_metric = sum(centered_results.values()) / len(centered_results)
print(json.dumps({"results": results, "centered_results": centered_results, "core_metric": core_metric}))
"""

# ---------------------------------------------------------------------------
# Worktree helpers
# ---------------------------------------------------------------------------

def create_worktree(commit, tmpdir):
    """Create a git worktree for the given commit. Returns worktree path."""
    worktree_path = os.path.join(tmpdir, "worktree")
    subprocess.run(
        ["git", "worktree", "add", "--detach", worktree_path, commit],
        cwd=REPO_ROOT, capture_output=True, check=True,
    )
    return worktree_path


def remove_worktree(worktree_path):
    """Remove a git worktree."""
    subprocess.run(
        ["git", "worktree", "remove", "--force", worktree_path],
        cwd=REPO_ROOT, capture_output=True,
    )


def run_subprocess(codebase_path, script, args, timeout=600):
    """Run a worker script in the given codebase directory and return parsed JSON."""
    result = subprocess.run(
        [VENV_PYTHON, "-c", script] + args,
        cwd=codebase_path, capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip().split("\n")[-1])
    for line in reversed(result.stdout.strip().split("\n")):
        try:
            return json.loads(line)
        except (json.JSONDecodeError, KeyError):
            continue
    raise RuntimeError(f"No JSON output found. stderr: {result.stderr[-200:]}")


# ---------------------------------------------------------------------------
# Eval mode runners
# ---------------------------------------------------------------------------

def run_sample(codebase_path, model_tag):
    """Run sample generation and return completion string."""
    data = run_subprocess(codebase_path, SAMPLE_WORKER_SCRIPT,
                          [model_tag, PROMPT, str(MAX_TOKENS)], timeout=120)
    return data["completion"]


def run_viz(codebase_path, model_tag):
    """Run attention visualization, saving images to experiments/<model_tag>/."""
    output_dir = os.path.join(REPO_ROOT, "experiments", model_tag)
    result = subprocess.run(
        [VENV_PYTHON, "-m", "experiments.visualize_attention",
         "--model-tag", model_tag, "--prompt", PROMPT, "--output-dir", output_dir],
        cwd=codebase_path, capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip().split("\n")[-1])
    return output_dir


def run_bench(codebase_path, model_tag, max_per_task):
    """Run CORE benchmark and return results dict."""
    return run_subprocess(codebase_path, CORE_WORKER_SCRIPT,
                          [model_tag, str(max_per_task)], timeout=1800)


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def append_json(path, entry):
    """Append an entry to a JSON list file."""
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(entry)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate experiment variants")
    parser.add_argument("--sample", action=argparse.BooleanOptionalAction, default=True,
                        help="Run text completions (default: on)")
    parser.add_argument("--viz", action="store_true", default=False,
                        help="Run attention visualization")
    parser.add_argument("--bench", action="store_true", default=False,
                        help="Run DCLM CORE benchmark")
    parser.add_argument("--max-per-task", type=int, default=100,
                        help="Max examples per CORE task (default: 100)")
    args = parser.parse_args()

    if not (args.sample or args.viz or args.bench):
        print("Nothing to do — enable at least one of --sample, --viz, --bench")
        sys.exit(1)

    modes = []
    if args.sample: modes.append("sample")
    if args.viz:    modes.append("viz")
    if args.bench:  modes.append("bench")
    print(f"Eval modes: {', '.join(modes)}")
    print(f"Prompt: {PROMPT!r}\n")

    samples_path = os.path.join(REPO_ROOT, "experiments", "eval_samples.json")
    bench_path = os.path.join(REPO_ROOT, "experiments", "eval_bench.json")

    for model_tag, commit, desc in EXPERIMENTS:
        label = f"{model_tag} @ {commit or 'HEAD'}"
        print(f"--- {label} ({desc}) ---")

        codebase_path = REPO_ROOT
        tmpdir_obj = None
        worktree_path = None

        try:
            # Set up worktree if needed
            if commit is not None:
                tmpdir_obj = tempfile.TemporaryDirectory()
                worktree_path = create_worktree(commit, tmpdir_obj.name)
                codebase_path = worktree_path

            # -- Sample mode --
            if args.sample:
                try:
                    completion = run_sample(codebase_path, model_tag)
                    print(f"  [sample] {completion}")
                    append_json(samples_path, {
                        "model_tag": model_tag, "commit": commit,
                        "desc": desc, "prompt": PROMPT, "completion": completion,
                    })
                except Exception as e:
                    print(f"  [sample] skipped: {e}")

            # -- Viz mode --
            if args.viz:
                try:
                    output_dir = run_viz(codebase_path, model_tag)
                    print(f"  [viz] saved to {output_dir}/")
                except Exception as e:
                    print(f"  [viz] skipped: {e}")

            # -- Bench mode --
            if args.bench:
                try:
                    bench_result = run_bench(codebase_path, model_tag, args.max_per_task)
                    core = bench_result["core_metric"]
                    print(f"  [bench] CORE={core:.4f}")
                    append_json(bench_path, {
                        "model_tag": model_tag, "commit": commit,
                        "core_metric": core, "task_results": bench_result["results"],
                    })
                except Exception as e:
                    print(f"  [bench] skipped: {e}")

        finally:
            if worktree_path:
                remove_worktree(worktree_path)
            if tmpdir_obj:
                tmpdir_obj.cleanup()

        print()


if __name__ == "__main__":
    main()
