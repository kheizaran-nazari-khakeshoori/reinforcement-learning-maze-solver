# Architecture



**High-Level Architecture**

The app is a layered RL system: Gymnasium environment exposes MDP, agents implement TD updates, and orchestration scripts handle training, evaluation, and rendering. All components communicate via discrete `state` integers and `Q(s,a)` tables, making agents swappable.

**System Data Flow**
```
┌───────────────────────┐
│   Grid Config / CLI   │  size, walls, traps, episodes
└───────────┬───────────┘
            │
            ▼
┌───────────────────────┐
│ MazeEnv (Gymnasium)   │  env.py:32  Discrete(25), Discrete(4)
└───────────┬───────────┘
            │
      ┌─────┼─────────────────┐
      ▼     ▼                 ▼
 [train.py] [run_episode]  [step() / reset()]
      │     │                 │
      └─────┼─────────────────┘
            ▼
┌───────────────────────┐
│  QLearning / Sarsa    │  agents.py  Q-table [25×4]
└───────────┬───────────┘
            │
      ┌─────┼────────────┐
      ▼     ▼            ▼
[evaluate] [bfs]   [MazeGUI / plot]
      │     │            │
      └─────┼────────────┘
            ▼
┌───────────────────────┐
│  Metrics + GIF        │  success, avg_steps, assets/*.gif
└───────────────────────┘
```

**Component Details**

***Core MDP & Agents***
**Location:** `maze_solver/env.py:32`, `maze_solver/agents.py:30`, `maze_solver/utils.py:3`
**Responsibilities:**
- State encoding `_pos_to_state` / `_state_to_pos`, validity and terminal checks
- Step/reward logic and Q-table TD updates
- BFS for ground-truth optimal path

***Orchestration & Visualization***
**Location:** `train.py:4`, `evaluate.py:3`, `compare.py:5`, `plot.py:6`, `maze_solver/gui.py:4`, `record.py:1`
**Responsibilities:**
- Episode loop, epsilon decay, greedy evaluation
- Q vs SARSA comparison, reward smoothing
- PyGame window, frame capture, ffmpeg pipe

***Technical Highlights***
- Custom Gymnasium env without external maze dependency
- Raw `rgb24` frame pipe to ffmpeg for reproducible video (Fedora-compatible `mpeg4` encoder)
- Deterministic `Discrete` spaces enable simple tabular convergence proof

---


