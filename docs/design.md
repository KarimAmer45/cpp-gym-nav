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
is intentionally a core primitive rather than a custom SB3 `VecEnv`; that adapter is the next step if
training measurements show the native batch materially reduces time-to-success.

## Realism increments

Each feature is opt-in and has a regression test proving that flags-off behavior matches the baseline:

- sensor noise combines seeded Gaussian noise, dropout, and quantization;
- actuator limits bound per-step changes in linear and angular velocity;
- action delay uses a fixed queue of up to two control steps;
- domain randomization jitters obstacles and adds seeded wheel slip.

The intended experimental workflow is hypothesis, one flagged change, regression test, then a
training/evaluation comparison over fixed held-out seeds.

## Unreal bridge boundary

An Unreal implementation should expose reset state, action stepping, observation, terminal reason,
and render frames using the same normalized schema. The first integration should use a versioned
message format and deterministic tick control. Shared memory can replace sockets only after measuring
that transport is significant. Unreal is intentionally not vendored into this MVP.
