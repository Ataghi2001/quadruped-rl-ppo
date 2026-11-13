# Evaluation Results

Model: `best_model.zip`
Episodes per agent: 20

| Metric | Random Agent | PPO Agent |
|---|---:|---:|
| Episode Reward | -25.79 +/- 35.82 | 1206.82 +/- 652.20 |
| Forward Distance (m) | 0.21 +/- 0.97 | 44.23 +/- 25.66 |
| Episode Length | 90.65 +/- 77.25 | 715.15 +/- 369.80 |

Values are mean +/- sample standard deviation.
Both agents use the same 20 reset seeds.
