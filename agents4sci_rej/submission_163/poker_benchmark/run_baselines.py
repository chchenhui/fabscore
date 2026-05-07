
from env import HUEnv
from deep_cfr import DeepCFR, obs_to_vec
from cfr import CFRTabular, MCCFRExternal
from nfsp import NFSP

def main():
    env = HUEnv()
    # CFR demo
    cfr = CFRTabular()
    cfr.iterate(env, T=1000)
    pi_cfr = cfr.average_policy()
    print("CFR states learned:", len(pi_cfr))

    # MCCFR skeleton
    mccfr = MCCFRExternal()
    mccfr.iterate(env, T=1000)
    print("MCCFR states learned:", len(mccfr.average_policy()))

    # DeepCFR skeleton
    d = DeepCFR()
    d.collect_traversals(env, n=2000)
    d.train(steps=10)
    obs = env.observe()
    print("DeepCFR action probs:", d.action_probs(obs))

    # NFSP skeleton (no training loop here; just API proof)
    nfsp = NFSP()
    print("NFSP avg action probs:", nfsp.avg_action_probs(obs_to_vec(obs)))

if __name__=="__main__":
    main()
