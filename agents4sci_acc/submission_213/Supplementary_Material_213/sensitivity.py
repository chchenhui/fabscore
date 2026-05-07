
import argparse, os, csv, json
from rl_env import EnvConfig, BuildingLifecycleEnv
from dqn_agent import DQNAgent, DQNConfig
import numpy as np

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
    parser.add_argument("--out", default="results/sensitivity_US.csv")
    args = parser.parse_args()

    base_env = BuildingLifecycleEnv(EnvConfig(case=args.case, data_path=args.data))
    agent = DQNAgent(DQNConfig(state_dim=base_env.state_dim(), action_dim=base_env.action_space(), gamma=base_env.gamma))
    agent.load(args.model)

    scenarios = [
        ("Baseline","SCC=base;Prod=base", 0.0, 0.0),
        ("Low Carbon Price","SCC=-0.737 (50 per ton vs base)", -0.737, 0.0),  # scale factor approx for US
        ("High Carbon Price","SCC=+0.579 (300 per ton vs base)", +0.579, 0.0),
        ("Low Productivity","Prod=-0.5x", 0.0, -0.5),
        ("High Productivity","Prod=+0.4286x (5% vs base 3.5%)", 0.0, +0.4286),
    ]

    rows = []
    for name, desc, scc_delta_frac, prod_delta_frac in scenarios:
        env = BuildingLifecycleEnv(EnvConfig(case=args.case, data_path=args.data))
        # Adjust parameters in env for scenario
        env.scc = env.scc * (1.0 + scc_delta_frac)
        env.productivity_gain_hq = env.productivity_gain_hq * (1.0 + prod_delta_frac)

        R, info = eval_greedy(env, agent)
        rows.append({
            "scenario": name,
            "description": desc,
            "reward_eval": R,
            "energy_cost_npv": info.get("energy_cost_npv", ""),
            "carbon_cost_npv": info.get("carbon_cost_npv", ""),
            "productivity_value_npv": info.get("productivity_value_npv", ""),
            "total_npv_costs": info.get("total_npv_costs",""),
            "total_benefits": info.get("total_benefits","")
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote sensitivity results to {args.out}")

if __name__ == "__main__":
    main()
