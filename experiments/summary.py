"""Print a concise summary table of all experiments."""
import json
import os

DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(DIR, "eval_bench.json")) as f:
    bench = json.load(f)
with open(os.path.join(DIR, "eval_samples.json")) as f:
    samples = json.load(f)

# Merge all experiment keys
all_tags = list(dict.fromkeys(list(bench.keys()) + list(samples.keys())))

rows = []
for tag in all_tags:
    desc = samples[tag]["desc"] if tag in samples else ""
    core = bench[tag]["core_metric"] if tag in bench else None
    completion = samples[tag]["completion"].strip() if tag in samples else ""
    rows.append((tag, desc, core, completion))

# Sort by core metric descending
rows.sort(key=lambda r: r[2] if r[2] is not None else -1, reverse=True)

# Truncate completions
max_comp = 80
for i, (tag, desc, core, completion) in enumerate(rows):
    if len(completion) > max_comp:
        completion = completion[:max_comp] + "..."
    rows[i] = (tag, desc, core, completion)

# Column widths
tag_w = max(len("Tag"), max(len(r[0]) for r in rows))
desc_w = max(len("Description"), max(len(r[1]) for r in rows))
core_w = max(len("CORE"), 6)
comp_w = max(len("Sample completion"), max(len(r[3]) for r in rows))

print("# Linear Attention Experiment Results")
print()
print('Prompt: "The capital of France is"')
print()
print(f"| {'Tag':<{tag_w}} | {'Description':<{desc_w}} | {'CORE':>{core_w}} | {'Sample completion':<{comp_w}} |")
print(f"|{'-' * (tag_w + 2)}|{'-' * (desc_w + 2)}|{'-' * (core_w + 2)}|{'-' * (comp_w + 2)}|")

for tag, desc, core, completion in rows:
    core_str = f"{core:.4f}" if core is not None else "n/a"
    print(f"| {tag:<{tag_w}} | {desc:<{desc_w}} | {core_str:>{core_w}} | {completion:<{comp_w}} |")
