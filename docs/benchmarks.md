# Reproducible benchmark and training results

Measured on 2026-08-01 from the repository's release build. The local host reported Windows 11
10.0.26200, Python 3.12.13, 10 logical processors, and `Intel64 Family 6 Model 165 Stepping 5`. The
portable validation build used Zig's Clang 21.1.0 because this host did not have the MSVC C++ workload
installed.

## Environment-step throughput

Command:

```bash
python train/benchmark.py --steps 100000 --vector-envs 16
```

| Python-visible configuration | Steps/s | Mean step (us) | p50 (us) | p95 (us) |
|---|---:|---:|---:|---:|
| Gymnasium wrapper, single environment | 19,268 | 37.95 | 37.40 | 39.40 |
| Gymnasium `SyncVectorEnv`, 16 environments | 21,110 | 46.18 | 45.84 | 48.84 |
| Native C++ batch, 16 environments | 860,916 | 0.71 | 0.70 | 0.78 |

The native C++ microbenchmark separately measured 2,607,250 steps/s (383.5 ns/step) over one million
steps. That gap identified Python crossings and wrapper bookkeeping as the dominant cost. The native
batch then raised Python-visible aggregate throughput by 44.7x relative to one Gymnasium environment.

The native batch result above is a core-throughput measurement, not an end-to-end SB3 training result.
It does not include Gymnasium time-limit bookkeeping and resets the full batch whenever any world
reaches a terminal state. The section below wires it into training.

## Training throughput: native vectorization

Command:

```bash
python train/benchmark.py --training
```

Raw environment throughput is not the same as faster RL. A custom SB3 `VecEnv` (`BatchWorldVecEnv`)
advances all 16 worlds in one GIL-released C++ call, and `train_ppo.py --native-vec 16` trains on it.
The same PPO recipe was trained three ways for 20k steps, with throughput measured over the steps SB3
actually collected:

| Training backend | Train steps/s | Speedup vs single |
|---|---:|---:|
| single Gymnasium env | 1,722 | 1.00x |
| SB3 `DummyVecEnv`, 16 envs (Python loop) | 5,602 | 3.25x |
| Native `BatchWorldVecEnv`, 16 envs | **7,518** | **4.37x** |

For a 200k-step run that is roughly 116 s down to 27 s of wall-clock. Two caveats a reviewer should
hear stated up front:

- About 3.25x of the speedup is generic vectorization (batched policy updates and amortized Python
  overhead) that any `VecEnv` provides. The native batched step contributes the remaining ~1.34x over
  SB3's Python-loop `DummyVecEnv`.
- The 44.7x raw-environment figure does not carry over to training. The policy update, not environment
  stepping, is the bottleneck here: raising lidar from 16 to 64 beams left training throughput
  essentially unchanged. The batched step matters most when the environment is expensive enough to
  dominate the loop.

## PPO versus random

Command:

```bash
python train/train_ppo.py --steps 200000 --eval-episodes 100
```

Both policies used the same 100 held-out episode seeds starting at 105951.

| Policy | Success | Collision | Mean return | Mean episode length |
|---|---:|---:|---:|---:|
| PPO | **72%** | 14% | **14.17** | 94.48 |
| Seeded random | 0% | 52% | -8.25 | 209.19 |

CPU-only training took 117.4 seconds. The original target of 80% held-out success was not reached in
this fixed 200k-step run, so no train-to-80 time is claimed. The exact JSON, learning curve, and demo
GIF are checked into `assets/generated`; the model and TensorBoard event files are reproducible but
git-ignored to keep the repository small.

## Robustness under domain randomization

Command:

```bash
python train/train_ppo.py --realism --steps 200000 --eval-episodes 100 \
  --output-dir assets/generated/realism
```

Enabling all four realism features at once (sensor noise, actuator limits, a one-step action delay, and
dynamics randomization) makes the task substantially harder. The same PPO recipe still learns, and the
resulting policy is scored both under the training perturbations and, over the same seeds, on clean
dynamics.

| Policy | Success | Collision | Mean return | Mean episode length |
|---|---:|---:|---:|---:|
| PPO under realism | **33%** | 38% | 5.05 | 143.11 |
| PPO transferred to clean | 35% | 46% | 3.11 | 129.65 |
| Seeded random under realism | 0% | 63% | -8.90 | 194.10 |

The clean specialist above reaches 72% on clean dynamics, so the domain-randomization policy trades
peak clean performance for behavior that is essentially unchanged whether or not the perturbations are
present — the expected robustness/optimality trade-off. Per-feature ablations are a natural next
experiment.
