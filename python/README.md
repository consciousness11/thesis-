# PPO + GAT sidecar for WorkflowSim

Python half of the bridge described in the plan
(`cozy-fluttering-leaf.md`).  The Java side
(`PPOSchedulingAlgorithm` + `PPOTrainerExample`) is a TCP client; this
module hosts the policy.

## One-time setup

PyTorch wheels are stable for Python 3.10–3.12.  The repo machine has
Python 3.14 on PATH, which is too new — make a 3.11 venv first.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r python/requirements.txt
```

## Smoke-test training (≈3000 timesteps, ~10 episodes per workflow)

Terminal 1 (Python server):

```
python python/train.py --port 7777 --total-timesteps 3000
```

Terminal 2 (Java client, from repo root, after `javac`):

```
java -Dppo.port=7777 -Dppo.episodes=10 ^
  -cp "build;lib/commons-math3-3.2.jar;lib/flanagan.jar;lib/jdom-2.0.0.jar" ^
  org.workflowsim.examples.scheduling.PPOTrainerExample
```

A checkpoint lands at `python/runs/ppo_smoke/ppo_smoke.zip`.

## Inference (single workflow, eval mode)

```
python python/infer.py --port 7778 --checkpoint python/runs/ppo_smoke/ppo_smoke.zip
```

```
java -Dppo.port=7778 ^
  -cp "build;lib/commons-math3-3.2.jar;lib/flanagan.jar;lib/jdom-2.0.0.jar" ^
  org.workflowsim.examples.scheduling.PPOSchedulingAlgorithmExample
```

## DQN vs PPO regression gate

```
java -cp "build;lib/commons-math3-3.2.jar;lib/flanagan.jar;lib/jdom-2.0.0.jar" ^
  org.workflowsim.examples.scheduling.DQNSchedulingAlgorithmExample > dqn.log
# ... run PPO infer + java client and pipe to ppo.log ...
python python/compare.py dqn.log ppo.log
```

`compare.py` exits 0 when PPO is within 2× of DQN on makespan AND energy
for every workflow.

## File map

| File                | Role                                                            |
| ------------------- | --------------------------------------------------------------- |
| `protocol.py`       | 4-byte-length-prefixed JSON frame I/O over a `socket.socket`    |
| `workflowsim_env.py`| Gymnasium env wrapping the TCP server; one Java client per env  |
| `gat_encoder.py`    | Raw-PyTorch masked 2-layer multi-head GAT                       |
| `policy.py`         | SB3 `BaseFeaturesExtractor` combining GAT(DAG), VM MLP, task MLP|
| `train.py`          | MaskablePPO smoke-test driver, checkpoint save                  |
| `infer.py`          | Load checkpoint, deterministic eval                             |
| `compare.py`        | DQN-vs-PPO delta table; non-zero exit if PPO > 2× DQN           |
