# Unreal Engine bridge

This is the "Unreal Engine + Gymnasium API" deliverable, built around the spec's
key idea: **the Python side never changes; the backend is swappable over a
socket.** The training code drives `NavEnvUnreal`, which speaks a versioned TCP
protocol. Today that protocol is answered by a reference server backed by the C++
core; an Unreal Engine process that answers the same protocol drops in without any
change to the RL code.

## What is implemented and tested here

- A versioned, newline-delimited JSON protocol over TCP (`backend_server.py`).
- `NavEnvUnreal(gymnasium.Env)` — the *identical* Gymnasium contract over the
  socket. It passes `gymnasium.utils.env_checker.check_env`.
- A reference backend server backed by the existing `NavEnv`/C++ core. With no
  host/port, `NavEnvUnreal` launches it in-process on an ephemeral port, so the
  bridge is self-contained for tests and CI.
- Tests (`tests/test_unreal_bridge.py`): `check_env`, **bit-exact parity** between
  a trajectory driven over the socket and the in-process env for the same
  seed/actions, and matching observation/action spaces.
- Verified that unchanged Stable-Baselines3 PPO code trains against the networked
  backend (`gym.make("CppGymNavUnreal-v0")`).

## What remains (the editor step)

Building and running an Unreal Engine 5 scene requires the UE editor (a GUI/GPU
application) and cannot be produced headlessly. The remaining work is a UE5
process that implements the protocol below and applies actions to a UE pawn /
reads observations from the UE world. The illustrative C++ stub at the end shows
the integration seam; the Python reference server is the authoritative contract to
match.

## Protocol (v1)

Transport: TCP, one JSON object per line (`\n`-delimited), UTF-8.

On connect the **server** sends a handshake:

```json
{"protocol_version": 1,
 "observation_low": [...], "observation_high": [...],
 "action_low": [-1, -1], "action_high": [1, 1]}
```

The **client** then sends requests and reads one response line per request:

| Request | Response |
|---|---|
| `{"cmd": "reset", "seed": <int\|null>}` | `{"observation": [...], "info": {...}}` |
| `{"cmd": "step", "action": [lin, ang]}` | `{"observation": [...], "reward": r, "terminated": b, "truncated": b, "info": {...}}` |
| `{"cmd": "close"}` | (connection closes) |

`action` is normalized to `[-1, 1]`; the backend scales it to physical speed
limits. `observation` is the 22-dim vector (6 navigation/body features + 16
normalized lidar beams). `terminated` = goal or collision; `truncated` = step
budget exceeded.

## Run the reference bridge

Standalone server (one process), client in another:

```bash
python -m cpp_gym_nav.backend_server --host 127.0.0.1 --port 8917
```

```python
from cpp_gym_nav import NavEnvUnreal

env = NavEnvUnreal("127.0.0.1", 8917)   # or NavEnvUnreal() to auto-launch the reference backend
obs, info = env.reset(seed=5951)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

Point `NavEnvUnreal(host, port)` at a running Unreal server instead, and the same
call sequence (and the same PPO training script) drives the UE scene.

## Unreal Engine 5 side (illustrative C++ stub)

Implement a listener that answers the protocol and bridges to a UE pawn. Add the
`Sockets`, `Networking`, and `Json` modules to your `*.Build.cs`. This sketch is a
reference for the integration seam, not a drop-in compilable file — the exact pawn
control and trace setup depend on your scene.

```cpp
// NavBridgeActor.h  (place a single instance in the level)
UCLASS()
class ANavBridgeActor : public AActor {
    GENERATED_BODY()
public:
    virtual void BeginPlay() override;      // open FTcpListener on the configured port
    virtual void Tick(float DeltaSeconds) override;

private:
    bool OnConnect(FSocket* Client, const FIPv4Endpoint& Endpoint);  // send handshake
    void ServiceRequest(const FString& Line);                        // parse + dispatch

    void ApplyAction(float Linear, float Angular);   // set pawn velocity from normalized action
    TArray<float> ReadObservation() const;           // goal-frame features + LineTrace lidar beams
    float StepReward(bool& bTerminated) const;       // potential progress - penalties (+ terminal)

    FTcpListener* Listener = nullptr;
    FSocket* Client = nullptr;
    int32 ElapsedSteps = 0;
    static constexpr int32 MaxEpisodeSteps = 300;
};

// NavBridgeActor.cpp (core of ServiceRequest)
void ANavBridgeActor::ServiceRequest(const FString& Line) {
    TSharedPtr<FJsonObject> Msg;
    const auto Reader = TJsonReaderFactory<>::Create(Line);
    if (!FJsonSerializer::Deserialize(Reader, Msg) || !Msg.IsValid()) return;

    const FString Cmd = Msg->GetStringField(TEXT("cmd"));
    TSharedRef<FJsonObject> Out = MakeShared<FJsonObject>();

    if (Cmd == TEXT("reset")) {
        const int64 Seed = Msg->HasTypedField<EJson::Number>(TEXT("seed"))
            ? (int64)Msg->GetNumberField(TEXT("seed")) : FMath::Rand();
        ResetWorld(Seed);                    // deterministic start/goal/obstacles from Seed
        ElapsedSteps = 0;
        Out->SetArrayField(TEXT("observation"), ToJson(ReadObservation()));
        Out->SetObjectField(TEXT("info"), MakeShared<FJsonObject>());
    } else if (Cmd == TEXT("step")) {
        const TArray<TSharedPtr<FJsonValue>> A = Msg->GetArrayField(TEXT("action"));
        ApplyAction(A[0]->AsNumber(), A[1]->AsNumber());
        // advance one fixed tick here (deterministic dt), then read results
        bool bTerminated = false;
        const float Reward = StepReward(bTerminated);
        const bool bTruncated = (++ElapsedSteps >= MaxEpisodeSteps) && !bTerminated;
        Out->SetArrayField(TEXT("observation"), ToJson(ReadObservation()));
        Out->SetNumberField(TEXT("reward"), Reward);
        Out->SetBoolField(TEXT("terminated"), bTerminated);
        Out->SetBoolField(TEXT("truncated"), bTruncated);
        Out->SetObjectField(TEXT("info"), MakeShared<FJsonObject>());
    } else if (Cmd == TEXT("close")) {
        Client->Close();
        return;
    }
    SendLine(Out);   // serialize Out to a single line + "\n"
}
```

Key correctness notes for a faithful UE backend:

- **Determinism:** seed a UE RNG from `reset`'s seed; use a fixed simulation tick
  so a seed + action sequence replays identically. This mirrors the C++ core's
  seeded `std::mt19937_64`.
- **Observation parity:** match the 22-dim layout and normalization exactly
  (robot-frame goal vector, distance, bearing/π, body velocities, then lidar via
  `LineTraceSingleByChannel` normalized by max range). Compare against the Python
  reference over a few seeds before trusting training results.
- **Action units:** the wire action is normalized `[-1, 1]`; scale to the pawn's
  speed limits on the UE side (as `BatchWorld.step` does).

## Recording the demo

With the UE server running, drive it from Python and screen-record the editor
viewport:

```python
python train/evaluate.py assets/generated/ppo_nav.zip   # after pointing evaluate at NavEnvUnreal(host, port)
```

A short clip of the UE scene being stepped from the unchanged Gymnasium/PPO loop
is the headline artifact — the same policy that solves the C++ env driving Unreal.

## Prior art referenced

- **Schola** (Unreal 5 RL plugin) and **Unreal ML Agents** — plugin-based bridges
  between UE and Python RL.
- **Colosseum** (the maintained AirSim fork) — a mature UE↔Python simulation
  bridge; its message framing and tick control informed this protocol.
- Epic's **Learning Agents** — a native in-engine RL alternative (PyTorch under the
  hood) that skips the socket entirely; a heavier but more integrated route.

The socket bridge here is deliberately the simplest faithful realization of
option 1 in the project spec: keep the physics/observation contract stable and
swap the backend process.
