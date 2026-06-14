"""Ablation sweep with Optuna for alpha/gamma/epsilon-decay."""

import itertools
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

from maze_solver.env import MazeEnv
from maze_solver.agents import QLearningAgent
from train import train
from evaluate import evaluate

SEARCH_SPACE = {
    "alpha": [0.05, 0.1, 0.2],
    "gamma": [0.9, 0.95, 0.99],
    "decay": [0.99, 0.995, 0.999],
}

def run_ablation(episodes=500, trials=10, seed=0):
    if HAS_OPTUNA:
        def objective(trial):
            alpha = trial.suggest_float("alpha", 0.05, 0.2)
            gamma = trial.suggest_float("gamma", 0.9, 0.99)
            decay = trial.suggest_float("decay", 0.99, 0.999)
            env = MazeEnv(size=5, seed=seed)
            agent = QLearningAgent(env.observation_space.n, env.action_space.n, alpha=alpha, gamma=gamma)
            train(agent, env, episodes=episodes, decay=decay, seed=seed)
            res = evaluate(agent, env, episodes=20, seed=seed)
            return res["success"]
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trials)
        return study.best_params, study.best_value
    else:
        # grid search fallback
        best=None
        best_params=None
        for alpha, gamma, decay in itertools.product(SEARCH_SPACE["alpha"], SEARCH_SPACE["gamma"], SEARCH_SPACE["decay"]):
            env = MazeEnv(size=5, seed=seed)
            agent = QLearningAgent(env.observation_space.n, env.action_space.n, alpha=alpha, gamma=gamma)
            train(agent, env, episodes=episodes, decay=decay, seed=seed)
            res = evaluate(agent, env, episodes=20, seed=seed)
            if best is None or res["success"] > best:
                best = res["success"]
                best_params = {"alpha":alpha,"gamma":gamma,"decay":decay}
        return best_params, best

if __name__ == "__main__":
    print(run_ablation())

# sweep supports alpha/gamma/decay
