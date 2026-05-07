
import argparse, os, csv, json, numpy as np
from rl_env import EnvConfig, BuildingLifecycleEnv
from dqn_agent import DQNAgent, DQNConfig

def eval_greedy(env, agent):
    s = env.reset()
    total_r = 0.0
    done = False
    while not done:
        with np.errstate(all='ignore'):
            a = int(np.argmax(agent.q.forward.__self__.net[-1].weight.detach().numpy() @ s + agent.q.forward.__self__.net[-1].bias.detach().numpy())) if hasattr(agent, 'q') else 0
        s, r, done, info = env.step(a)
        total_r += r
    return total_r, info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["US","UK"], default="US")
    parser.add_argument("--data", default="data_sources.csv")
    parser.add_argument("--model", default="models/dqn_US.pt")
    parser.add_argument("--out", default="results/robustness_US.csv")
    args = parser.parse_args()

    base_env = BuildingLifecycleEnv(EnvConfig(case=args.case, data_path=args.data))
    agent = DQNAgent(DQNConfig(state_dim=base_env.state_dim(), action_dim=base_env.action_space(), gamma=base_env.gamma))
    agent.load(args.model)

    tests = []
    # Baseline
    env = BuildingLifecycleEnv(EnvConfig(case=args.case, data_path=args.data))
    R0, info0 = eval_greedy(env, agent)
    tests.append({"condition":"Baseline","reward_eval":R0, **info0})

    # ±20% parameter noise on eui and elec price
    env = BuildingLifecycleEnv(EnvConfig(case=args.case, data_path=args.data))
    env.baseline_eui *= 1.2
    env.elec_price *= 1.2
    R1, info1 = eval_greedy(env, agent)
    tests.append({"condition":"+20% Noise","reward_eval":R1, **info1})

    env = BuildingLifecycleEnv(EnvConfig(case=args.case, data_path=args.data))
    env.baseline_eui *= 0.8
    env.elec_price *= 0.8
    R2, info2 = eval_greedy(env, agent)
    tests.append({"condition":"-20% Noise","reward_eval":R2, **info2})

    # +2C climate scenario -> emulate higher cooling needs via energy_mult
    env = BuildingLifecycleEnv(EnvConfig(case=args.case, data_path=args.data))
    env.baseline_eui *= 1.1
    R3, info3 = eval_greedy(env, agent)
    tests.append({"condition":"+2C Climate Scenario (~+10% base EUI)","reward_eval":R3, **info3})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    keys = sorted(set(k for d in tests for k in d.keys()))
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(tests)
    print(f"Wrote robustness results to {args.out}")

if __name__ == "__main__":
    main()
