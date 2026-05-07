
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, List, Optional

# Simple, dependency-light RL environment for the sustainable office lifecycle.
# States: [stage_one_hot(4), design_feature_one_hot(3), construction_feature_one_hot(3),
#          eui_norm, embodied_carbon_norm, grid_ci_norm, elec_price_norm]
# Actions per stage:
#   stage 0/1 (Pre-design/Design): 0=Conventional, 1=Green, 2=Ultra
#   stage 2 (Construction):        0=Standard, 1=GreenMaterials, 2=SocialProcure
#   stage 3 (Operation):           0=StandardFM, 1=SmartEnergy, 2=WellnessProgram
#
# Rewards combine -NPV_cost + alpha_i * social/env values (scaled via monetization).
# Noise: energy outcomes have +/- energy_noise fraction.
#
# Load parameters from data_sources.csv for a given case: "US" or "UK".

@dataclass
class EnvConfig:
    case: str = "US"
    data_path: str = "data_sources.csv"
    alpha_carbon: float = 1.0  # multiplier for SCC monetization, can be tuned
    alpha_productivity: float = 1.0
    alpha_jobs: float = 0.0  # set to >0 to monetize job-years (optional)
    seed: int = 42

@dataclass
class StateVars:
    stage: int = 0  # 0 Pre-design, 1 Design, 2 Construction, 3 Operation
    design_choice: int = -1   # -1 unset, else 0/1/2
    constr_choice: int = -1   # -1 unset, else 0/1/2
    eui: float = 0.0
    embodied_carbon: float = 0.0
    grid_ci: float = 0.0
    elec_price: float = 0.0

class BuildingLifecycleEnv:
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.params = self._load_params(cfg.case, cfg.data_path)

        # Derived constants
        self.gfa = self._get("building_gfa")  # m2
        self.operation_years = int(self._get("operation_years"))
        self.discount_rate = self._get("discount_rate")
        self.gamma = self._get("gamma")
        self.energy_noise = self._get("energy_noise")

        # baselines
        self.baseline_eui = self._get("baseline_eui")
        self.high_perf_eui = self._get("high_perf_eui")
        self.grid_ci = self._get("grid_carbon_intensity")
        self.elec_price = self._get("electricity_price")
        self.capex_per_m2 = self._get("construction_cost")
        self.embodied_baseline = self._get("embodied_carbon_baseline")
        self.embodied_green_frac = self._get("embodied_carbon_reduction_green")
        self.design_prem_green = self._get("design_premium_green")
        self.design_prem_ultra = self._get("design_premium_ultra")

        self.scc = self._get("scc")
        self.productivity_gain_hq = self._get("productivity_gain_hq_ieq")
        self.occupants = self._get("occupants_per_10000m2")
        self.avg_salary = self._get("avg_salary")
        self.value_1pct_prod = self._get("value_of_1pct_productivity")

        self.reset()

    def _load_params(self, case: str, path: str) -> Dict[str, float]:
        df = pd.read_csv(path)
        sub = df[df["case"] == case]
        if sub.empty:
            raise ValueError(f"No params for case {case}")
        params = {}
        for _, row in sub.iterrows():
            key = row["key"]
            try:
                val = float(row["value"])
            except Exception:
                val = row["value"]
            params[key] = val
        return params

    def _get(self, key: str) -> float:
        if key not in self.params:
            raise KeyError(f"Missing parameter: {key}")
        return float(self.params[key])

    def reset(self) -> np.ndarray:
        self.state = StateVars(stage=1,  # start at Design stage for simplicity
                               design_choice=-1,
                               constr_choice=-1,
                               eui=self.baseline_eui,
                               embodied_carbon=self.embodied_baseline,
                               grid_ci=self.grid_ci,
                               elec_price=self.elec_price)
        return self._obs()

    def _obs(self) -> np.ndarray:
        # One-hot stage
        stage_oh = np.zeros(4)
        stage_oh[int(self.state.stage)] = 1.0

        # One-hot choices
        design_oh = np.zeros(3)
        if self.state.design_choice >= 0:
            design_oh[self.state.design_choice] = 1.0
        constr_oh = np.zeros(3)
        if self.state.constr_choice >= 0:
            constr_oh[self.state.constr_choice] = 1.0

        # Normalize continuous features by simple constants
        eui_norm = self.state.eui / max(1e-6, self.baseline_eui)
        emb_norm = self.state.embodied_carbon / max(1e-6, self.embodied_baseline)
        gci_norm = self.state.grid_ci / max(1e-6, self.grid_ci)
        ep_norm = self.state.elec_price / max(1e-6, self.elec_price)

        return np.concatenate([stage_oh, design_oh, constr_oh,
                               [eui_norm, emb_norm, gci_norm, ep_norm]]).astype(np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        reward = 0.0
        info = {}
        done = False

        if self.state.stage == 1:  # Design
            # Apply design choice effects
            self.state.design_choice = action  # 0/1/2
            if action == 0:
                # conventional
                design_prem = 0.0
                self.state.eui = self.baseline_eui
                prod_gain = 0.0
            elif action == 1:
                # green
                design_prem = self.design_prem_green
                self.state.eui = self.high_perf_eui + 0.25*(self.baseline_eui-self.high_perf_eui)  # mid
                prod_gain = 0.5*self.productivity_gain_hq
            elif action == 2:
                # ultra
                design_prem = self.design_prem_ultra
                self.state.eui = self.high_perf_eui
                prod_gain = self.productivity_gain_hq
            else:
                raise ValueError("Invalid design action")
            capex = self.capex_per_m2 * self.gfa * (1.0 + design_prem)
            # Negative immediate cash outflow
            reward += -capex  # note: raw cash, NPV handled approximately later during operation
            info.update({"capex": capex, "prod_gain_design": prod_gain})

            self.state.stage = 2

        elif self.state.stage == 2:  # Construction
            self.state.constr_choice = action
            # Embodied carbon effect & social procurement
            if action == 0:
                emb = self.embodied_baseline * self.gfa
                job_multiplier = self._get("job_creation_baseline")
            elif action == 1:
                emb = (1.0 - self.embodied_green_frac) * self.embodied_baseline * self.gfa
                job_multiplier = self._get("job_creation_baseline")
            elif action == 2:
                emb = self.embodied_baseline * self.gfa
                job_multiplier = self._get("job_creation_enhanced")
            else:
                raise ValueError("Invalid construction action")
            self.state.embodied_carbon = emb / self.gfa  # back to per m2
            # monetize embodied carbon
            scc = self.scc
            # Convert kg -> tons
            emb_tons = emb / 1000.0
            carbon_cost = emb_tons * scc
            reward += -carbon_cost * self.cfg.alpha_carbon

            # jobs value (optional monetization example)
            capex_total_million = (self.capex_per_m2 * self.gfa) / 1e6
            job_years = job_multiplier * capex_total_million
            jobs_value = self.cfg.alpha_jobs * job_years  # if alpha_jobs=0, it's 0
            reward += jobs_value
            info.update({"embodied_carbon_total_kg": emb, "carbon_cost": carbon_cost,
                         "job_years": job_years, "jobs_value": jobs_value})

            self.state.stage = 3

        elif self.state.stage == 3:  # Operation (single aggregated step representing NPV over 20 years)
            # Operation actions affect realized energy and productivity programs
            energy_mult = 1.0
            prod_program_gain = 0.0
            if action == 0:
                pass
            elif action == 1:
                # smart energy management
                energy_mult = 0.9
            elif action == 2:
                # wellness program
                prod_program_gain = 0.5*self.productivity_gain_hq  # additional gain
            else:
                raise ValueError("Invalid operation action")

            eui_realized = self.state.eui * energy_mult
            # Add stochasticity
            noise = self.rng.uniform(-self.energy_noise, self.energy_noise)
            eui_realized *= (1.0 + noise)

            # Energy and cost
            annual_kwh = eui_realized * self.gfa
            annual_cost = annual_kwh * self.elec_price
            # NPV of 20-year operation (years 2-21 => 20 years)
            npv_factor = sum([(1.0 / ((1.0 + self.discount_rate) ** (t+1))) for t in range(self.operation_years)])
            energy_cost_npv = annual_cost * npv_factor

            # Operational carbon
            annual_tons = (annual_kwh * self.grid_ci) / 1000.0  # kg->tons
            carbon_cost_npv = annual_tons * self.scc * npv_factor

            # Productivity valuation (design+program gains)
            # Approximate total productivity % = design effect + program effect, capped
            prev_design_gain = 0.0
            # Recover design gain from design_choice (consistent with step at stage 1)
            if self.state.design_choice == 1:
                prev_design_gain = 0.5*self.productivity_gain_hq
            elif self.state.design_choice == 2:
                prev_design_gain = self.productivity_gain_hq
            total_prod_gain = min(prev_design_gain + prod_program_gain, 0.15)  # cap at 15%

            occupants = self.occupants
            # Monetize productivity: value_of_1pct_productivity * percent * occupants * NPV factor
            value_per_pct = self.value_1pct_prod
            productivity_value_npv = occupants * value_per_pct * (total_prod_gain*100.0) * npv_factor

            # Reward aggregation
            total_npv_costs = energy_cost_npv + carbon_cost_npv * self.cfg.alpha_carbon
            total_benefits = self.cfg.alpha_productivity * productivity_value_npv
            reward += -total_npv_costs + total_benefits

            info.update({
                "eui_realized": eui_realized,
                "annual_kwh": annual_kwh,
                "annual_energy_cost": annual_cost,
                "energy_cost_npv": energy_cost_npv,
                "annual_tonCO2e": annual_tons,
                "carbon_cost_npv": carbon_cost_npv,
                "productivity_value_npv": productivity_value_npv,
                "total_npv_costs": total_npv_costs,
                "total_benefits": total_benefits
            })

            done = True

        else:
            raise ValueError("Unknown stage")

        obs = self._obs()
        return obs, float(reward), bool(done), info

    def action_space(self) -> int:
        # 3 actions valid at each stage
        return 3

    def state_dim(self) -> int:
        return len(self._obs())

