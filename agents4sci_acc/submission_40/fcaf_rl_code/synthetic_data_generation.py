"""
Synthetic dataset generation for FCAF-RL demonstration.

This script produces a CSV file containing synthetic patient-week records for
Medicaid care management. The synthetic data approximate the marginal
distributions of demographic and clinical variables observed in Medicaid
programmes.  Three states (Washington, Virginia and Ohio) are included.

The generated dataset includes the following columns:
    - patient_id: unique identifier for the patient
    - week: time index (0–25)
    - state: one of {WA, VA, OH}
    - age: integer age in years (20–80)
    - sex: categorical sex (Female or Male)
    - race: categorical race/ethnicity (White, Black, Hispanic, Other)
    - comorbidity_count: number of chronic conditions (0–6)
    - prior_ed_visits: number of emergency department visits in the prior year
    - mental_health_flag: binary indicator of mental health diagnosis
    - substance_use_flag: binary indicator of substance use disorder
    - social_need_score: continuous score (0–1) representing unmet social needs
    - intervention: one of nine interventions or "None" if no intervention
    - acute_event: binary indicator of an acute event in the following week

Usage:
    python synthetic_data_generation.py --output synthetic_data.csv --n_patients 5000

The default settings generate trajectories for 5,000 patients and 26 weeks per
patient (approximately 130,000 rows). Adjust --n_patients to increase or
decrease the dataset size. The script writes the synthetic dataset to the
specified CSV file.
"""
import argparse
import numpy as np
import pandas as pd


INTERVENTIONS = [
    "SubstanceUseSupport",
    "MentalHealthSupport",
    "ChronicConditionManagement",
    "FoodAssistance",
    "HousingAssistance",
    "TransportationAssistance",
    "UtilitiesAssistance",
    "ChildcareAssistance",
    "WatchfulWaiting",
    "None",
]


def generate_patient_data(patient_id: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate a synthetic trajectory for a single patient.

    Args:
        patient_id: integer identifier for the patient.
        rng: NumPy random generator.

    Returns:
        DataFrame with weekly observations for the patient.
    """
    n_weeks = rng.integers(12, 26)  # simulate 3–6 month episodes
    state = rng.choice(["WA", "VA", "OH"])
    age = int(rng.normal(loc=50, scale=15))
    age = int(np.clip(age, 20, 80))
    sex = rng.choice(["Female", "Male"])
    race = rng.choice(["White", "Black", "Hispanic", "Other"], p=[0.55, 0.25, 0.12, 0.08])
    comorbidity_count = int(rng.poisson(lam=2))
    comorbidity_count = int(np.clip(comorbidity_count, 0, 6))
    prior_ed_visits = int(rng.poisson(lam=1))
    mental_health_flag = int(rng.binomial(1, 0.2))
    substance_use_flag = int(rng.binomial(1, 0.1))
    social_need_score = float(rng.beta(a=2, b=5))  # skewed toward lower needs

    records = []
    for week in range(n_weeks):
        # intervention assignment: random with slight bias based on social need score
        if social_need_score > 0.7 and rng.random() < 0.3:
            intervention = rng.choice(INTERVENTIONS[:-1])  # choose one of nine interventions
        else:
            intervention = "None"

        # acute event probability depends on comorbidities, prior ED visits and intervention
        base_rate = 0.05 + 0.02 * comorbidity_count + 0.01 * prior_ed_visits
        if intervention != "None":
            # interventions reduce risk moderately
            base_rate *= 0.7
        base_rate = min(max(base_rate, 0.01), 0.8)
        acute_event = int(rng.random() < base_rate)

        records.append(
            {
                "patient_id": patient_id,
                "week": week,
                "state": state,
                "age": age,
                "sex": sex,
                "race": race,
                "comorbidity_count": comorbidity_count,
                "prior_ed_visits": prior_ed_visits,
                "mental_health_flag": mental_health_flag,
                "substance_use_flag": substance_use_flag,
                "social_need_score": social_need_score,
                "intervention": intervention,
                "acute_event": acute_event,
            }
        )
        # update risk factors for next week
        prior_ed_visits += acute_event
        social_need_score = float(np.clip(social_need_score + rng.normal(0, 0.05), 0.0, 1.0))

    return pd.DataFrame.from_records(records)


def generate_dataset(n_patients: int, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic dataset for multiple patients.

    Args:
        n_patients: number of patient trajectories to generate.
        seed: random seed for reproducibility.

    Returns:
        Concatenated DataFrame of all patient-week records.
    """
    rng = np.random.default_rng(seed)
    data_frames = []
    for patient_id in range(n_patients):
        df = generate_patient_data(patient_id, rng)
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Medicaid trajectories.")
    parser.add_argument("--output", type=str, default="synthetic_data.csv", help="Output CSV filename")
    parser.add_argument("--n_patients", type=int, default=5000, help="Number of patient trajectories to generate")
    args = parser.parse_args()
    df = generate_dataset(args.n_patients)
    df.to_csv(args.output, index=False)
    print(f"Synthetic dataset with {len(df)} rows written to {args.output}")


if __name__ == "__main__":
    main()