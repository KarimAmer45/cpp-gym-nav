# Design decisions, trade-offs, and safety notes

## Stable contract, replaceable backend

`NavEnv` owns the Gymnasium contract while `_core.World` owns simulation. Training code therefore
depends on `reset` and `step`, not on simulator details. A future Unreal adapter can implement the
same backend boundary over sockets or shared memory without changing PPO code.

## Observation and action design

The two continuous actions are body-frame linear and angular velocity commands. Continuous control
matches mobile-robot controllers and avoids imposing an arbitrary discrete motion vocabulary. The
observation uses robot-frame goal features so the policy does not need to learn rotational
invariance. All values are finite `float32` in `[-1, 1]`; lidar is normalized by maximum range.

The policy-facing action is normalized `[-1, 1]`. Two core entry points differ deliberately and are
documented in their bindings: `World.step` takes physical-unit actions (it is the single-world
primitive), while `BatchWorld.step` takes normalized actions and scales them, because it is driven by
the RL policy through the vectorized adapter. The Gymnasium wrapper always presents the normalized form.

## Kinematics integration

The unicycle model uses semi-implicit (symplectic) Euler: the heading is advanced first and the body
then translates along the updated heading. This is the standard choice for differential-drive
kinematics and is more stable than explicit Euler, which would translate along the stale pre-rotation
heading and bias curved motion.

## Reward and termination

The dense term is the one-step decrease in Euclidean goal distance. It is potential-based: returning
to an earlier position cancels earlier progress, so circling cannot accumulate progress reward. A
small time cost discourages stalling and a small effort cost discourages saturated commands. Goal and
collision are true terminal states; exhausting the step budget is truncation. Their separation is
important for correct value-function bootstrapping.

Known risks include policies exploiting collision geometry, oscillating around local obstacles, or
overfitting deterministic layouts. Regression tests cover action bounds and deterministic replay;
evaluation should additionally report collision and success rates over held-out seeds.

## Determinism and traceability

The seed passed to Gymnasium is forwarded to a per-world `std::mt19937_64`. It controls poses,
obstacle randomization, lidar noise, dropout, and slip. A seed plus action sequence exactly reproduces
a trajectory on the same build. The info dictionary exposes success, collision, distance, command
clipping, and the command actually applied.

## Hot path

The C++ world preallocates observation and step-result buffers. Obstacles and delayed actions use
fixed-size arrays. Python enters the core once per step, and the binding releases the GIL during
simulation. The returned NumPy observation is copied deliberately: exposing an internal mutable
buffer would let a later step silently mutate an observation retained by an RL replay buffer.

`SyncVectorEnv` provides a compatibility baseline but does not remove Python crossings. The native
`BatchWorld` advances K preallocated worlds per binding call and returns contiguous NumPy arrays. It
backs a custom SB3 `VecEnv` (`BatchWorldVecEnv`, used by `train_ppo.py --native-vec`) that handles
per-world auto-reset and truncation. Measured PPO training throughput is ~4.4x a single environment and
~1.3x SB3's Python-loop `DummyVecEnv`; the batched step removes the environment as a scaling factor, but
the honest end-to-end gain is bounded because the policy update, not stepping, dominates this small-MLP
loop (raising lidar beams 16 to 64 barely changed it). The batch matters most when the environment is
expensive enough to dominate.

## Realism increments

Each feature is opt-in and has a regression test proving that flags-off behavior matches the baseline:

- sensor noise combines seeded Gaussian noise, dropout, and quantization;
- actuator limits bound per-step changes in linear and angular velocity;
- action delay uses a fixed queue of up to two control steps;
- domain randomization adds seeded wheel slip (and jitters the fixed reference map when random
  layouts are disabled).

Separately, obstacle layouts are randomized per episode by default (`random_obstacles`, disable with
`--fixed-obstacles`), so the environment is a distribution of tasks rather than one memorized map.
The intended experimental workflow is hypothesis, one flagged change, regression test, then a
training/evaluation comparison over fixed held-out seeds. Reported success rates (66% on randomized
layouts, 72% on the fixed map) use default PPO hyperparameters; entropy, learning-rate, and GAE tuning
or a larger step budget are untapped headroom rather than a ceiling.

## Unreal bridge boundary

The bridge is implemented as a versioned, newline-delimited JSON protocol over TCP. `NavEnvUnreal`
satisfies the identical Gymnasium contract over the socket, and a reference server backed by the C++
core answers the protocol, so the same PPO/`check_env` code runs unchanged against a networked backend.
Tests assert bit-exact parity between the socket path and the in-process env, which is the guarantee a
real backend must preserve. An Unreal process becomes a drop-in replacement by answering the same
protocol — exposing reset, action stepping, observation, and terminal reason under the same normalized
schema with deterministic tick control. Shared memory can replace the socket only after measuring that
transport is a significant cost. Unreal itself is intentionally not vendored; the protocol and a UE5
C++ integration stub live in [unreal.md](unreal.md).
