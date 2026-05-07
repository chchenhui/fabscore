
import argparse, os, json, csv
import numpy as np
from rl_env import EnvConfig, BuildingLifecycleEnv
from dqn_agent import DQNAgent, DQNConfig

def run_episode(env, agent, greedy: bool=False):
    s = env.reset()
    total_r = 0.0
    traj = []
    done = False
    while not done:
        if greedy:
            # force epsilon=0 selection
            a = np.argmax(agent.q_target.forward.__self__.net[-1].weight.detach().numpy() @ s + agent.q_target.forward.__self__.net[-1].bias.detach().numpy()) if hasattr(agent, 'q_target') else 0
        else:
            a = agent.select_action(s)
        s2, r, done, info = env.step(a)
        traj.append({"stage": int(np.argmax(s[:4])), "action": int(a), **info})
        total_r += r
        if not greedy:
            agent.remember(s, a, r, s2, float(done))
            agent.update()
        s = s2
    return total_r, traj

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["US","UK"], default="US")
    parser.add_argument("--data", default="data_sources.csv")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--modeldir", default="models")
    args = parser.parse_args()

    env = BuildingLifecycleEnv(EnvConfig(case=args.case, data_path=args.data, seed=args.seed))
    cfg = DQNConfig(state_dim=env.state_dim(), action_dim=env.action_space(), gamma=env.gamma, seed=args.seed)
    agent = DQNAgent(cfg)

    # Training
    best_r = -1e30
    for ep in range(1, args.episodes+1):
        R, _ = run_episode(env, agent, greedy=False)
        if R > best_r:
            best_r = R
        if ep % 500 == 0:
            print(f"[{args.case}] Episode {ep}/{args.episodes} R={R:.2f}, best={best_r:.2f} eps={agent.epsilon:.2f}")

    os.makedirs(args.modeldir, exist_ok=True)
    model_path = os.path.join(args.modeldir, f"dqn_{args.case}.pt")
    agent.save(model_path)

    # Greedy evaluation and export trajectory
    R_eval, traj = run_episode(env, agent, greedy=True)
    os.makedirs(args.outdir, exist_ok=True)
    traj_path = os.path.join(args.outdir, f"{args.case}_trajectory.csv")
    with open(traj_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(traj[0].keys()))
        writer.writeheader()
        writer.writerows(traj)

    summary = {
        "case": args.case,
        "reward_eval": R_eval,
        "trajectory_csv": traj_path,
        "model_path": model_path,
        "episodes": args.episodes,
        "seed": args.seed
    }
    with open(os.path.join(args.outdir, f"{args.case}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved trajectory to {traj_path}")
    print(f"Eval reward: {R_eval:.2f}")

if __name__ == "__main__":
    main()
