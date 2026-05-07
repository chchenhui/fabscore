"""
Patient flow modeling for pharmaceutical forecasting.
This is how IQVIA and ZS actually model drug uptake.
Following Linus principle: Model the actual patient journey, not abstract diffusion.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class PatientFlowParams:
    """Parameters for patient flow model."""
    
    # Market dynamics
    eligible_patients: int
    annual_incidence: float  # New patients per year as % of eligible
    annual_discontinuation: float  # % who stop treatment each year
    
    # Adoption curve
    peak_penetration: float  # Maximum % of eligible on drug
    years_to_peak: int  # Time to reach peak
    
    # Treatment dynamics  
    adherence_rate: float  # % compliant with treatment
    switching_in_rate: float  # % switching from competitors
    switching_out_rate: float  # % switching to competitors
    
    # Pricing
    annual_price: float  # List price per patient per year
    net_to_gross: float  # GTN ratio
    
    # Access
    coverage_evolution: List[float]  # % with coverage by year
    prior_auth_approval: float  # % of PA requests approved


class PatientFlowModel:
    """
    Model drug revenue through explicit patient flows.
    This is the industry standard approach.
    """
    
    def __init__(self):
        self.params = None
        self.patient_states = None
    
    def set_params_from_drug(self, drug_row: pd.Series) -> PatientFlowParams:
        """
        Derive patient flow parameters from drug characteristics.
        
        Args:
            drug_row: Row from launches.parquet
        
        Returns:
            PatientFlowParams object
        """
        
        # Base parameters
        eligible = drug_row['eligible_patients_at_launch']
        annual_price = drug_row['list_price_month_usd_launch'] * 12
        gtn = drug_row['net_gtn_pct_launch']
        
        # Estimate incidence by therapeutic area
        ta_incidence = {
            'Oncology': 0.15,  # High incidence
            'Immunology': 0.08,
            'Cardiovascular': 0.10,
            'Endocrinology': 0.12,
            'Respiratory': 0.09,
            'Neurology': 0.06,  # Low incidence
            'Rare Disease': 0.03,  # Very low incidence
        }
        incidence = ta_incidence.get(drug_row.get('therapeutic_area', ''), 0.10)
        
        # Discontinuation rates by TA
        ta_discontinuation = {
            'Oncology': 0.40,  # High discontinuation
            'Immunology': 0.25,
            'Cardiovascular': 0.20,  # Chronic, lower discontinuation
            'Endocrinology': 0.15,
            'Respiratory': 0.22,
            'Neurology': 0.30,
            'Rare Disease': 0.10,  # Low discontinuation in rare
        }
        discontinuation = ta_discontinuation.get(drug_row.get('therapeutic_area', ''), 0.25)
        
        # Peak penetration based on competition and efficacy
        base_penetration = 0.30  # 30% base case
        
        # Adjust for competition
        competitors = drug_row.get('competitor_count_at_launch', 3)
        if competitors == 0:
            competition_mult = 1.5
        elif competitors <= 2:
            competition_mult = 1.0
        elif competitors <= 5:
            competition_mult = 0.7
        else:
            competition_mult = 0.4
        
        # Adjust for efficacy
        efficacy = drug_row.get('clinical_efficacy_proxy', 0.7)
        efficacy_mult = 0.5 + efficacy
        
        peak_penetration = base_penetration * competition_mult * efficacy_mult
        peak_penetration = min(peak_penetration, 0.60)  # Cap at 60%
        
        # Years to peak by TA
        ta_years_to_peak = {
            'Oncology': 3,  # Fast uptake
            'Immunology': 4,
            'Cardiovascular': 5,  # Slow uptake
            'Endocrinology': 4,
            'Respiratory': 4,
            'Neurology': 6,  # Very slow
            'Rare Disease': 2,  # Fast in rare
        }
        years_to_peak = ta_years_to_peak.get(drug_row.get('therapeutic_area', ''), 4)
        
        # Adherence by TA
        ta_adherence = {
            'Oncology': 0.75,  # High adherence in oncology
            'Immunology': 0.70,
            'Cardiovascular': 0.65,
            'Endocrinology': 0.60,
            'Respiratory': 0.68,
            'Neurology': 0.55,
            'Rare Disease': 0.80,  # High in rare disease
        }
        adherence = ta_adherence.get(drug_row.get('therapeutic_area', ''), 0.65)
        
        # Switching rates based on competition
        if competitors <= 1:
            switching_in = 0.15  # High switching in with low competition
            switching_out = 0.05  # Low switching out
        elif competitors <= 3:
            switching_in = 0.10
            switching_out = 0.10
        else:
            switching_in = 0.05  # Low switching in with high competition
            switching_out = 0.15  # High switching out
        
        # Coverage evolution based on access tier
        tier = drug_row.get('access_tier_at_launch', 'PA')
        if tier == 'OPEN':
            coverage = [0.80, 0.85, 0.90, 0.92, 0.95]  # Fast coverage growth
            pa_approval = 0.95  # High approval
        elif tier == 'PA':
            coverage = [0.40, 0.55, 0.65, 0.70, 0.75]  # Moderate coverage
            pa_approval = 0.60  # Moderate approval
        else:  # NICHE
            coverage = [0.15, 0.20, 0.25, 0.30, 0.35]  # Limited coverage
            pa_approval = 0.30  # Low approval
        
        # Extend coverage to more years if needed
        while len(coverage) < 10:
            coverage.append(coverage[-1])
        
        return PatientFlowParams(
            eligible_patients=eligible,
            annual_incidence=incidence,
            annual_discontinuation=discontinuation,
            peak_penetration=peak_penetration,
            years_to_peak=years_to_peak,
            adherence_rate=adherence,
            switching_in_rate=switching_in,
            switching_out_rate=switching_out,
            annual_price=annual_price,
            net_to_gross=gtn,
            coverage_evolution=coverage,
            prior_auth_approval=pa_approval
        )
    
    def simulate_patient_flow(self, params: PatientFlowParams, 
                            years: int = 5) -> Dict[str, np.ndarray]:
        """
        Simulate patient flow over time.
        
        Args:
            params: Patient flow parameters
            years: Number of years to simulate
        
        Returns:
            Dict with patient counts by state over time
        """
        
        # Initialize patient states with proper mass conservation
        states = {
            'eligible': np.zeros(years),
            'on_drug': np.zeros(years),
            'discontinued': np.zeros(years),
            'competitor_drug': np.zeros(years),  # Track switching to competitors
            'new_patients': np.zeros(years),
            'switching_in': np.zeros(years),
            'switching_out': np.zeros(years)
        }
        
        # Initial eligible population
        states['eligible'][0] = params.eligible_patients
        
        # Simulate each year
        for t in range(years):
            # Update eligible population (grows with incidence)
            if t > 0:
                new_eligible = states['eligible'][t-1] * params.annual_incidence
                states['eligible'][t] = states['eligible'][t-1] + new_eligible
            
            # Calculate target patients based on adoption curve
            if t < params.years_to_peak:
                # S-curve adoption to peak
                progress = t / params.years_to_peak
                target_penetration = params.peak_penetration * (
                    3 * progress**2 - 2 * progress**3  # Smooth S-curve
                )
            else:
                target_penetration = params.peak_penetration
            
            target_patients = states['eligible'][t] * target_penetration
            
            # Apply coverage and access constraints
            coverage = params.coverage_evolution[min(t, len(params.coverage_evolution)-1)]
            accessible_patients = target_patients * coverage * params.prior_auth_approval
            
            # Calculate flows
            if t == 0:
                # Launch year - all new starts
                states['new_patients'][t] = accessible_patients * 0.1  # Slow start
                states['on_drug'][t] = states['new_patients'][t]
            else:
                # Continuing patients (minus discontinuation)
                continuing = states['on_drug'][t-1] * (1 - params.annual_discontinuation)
                
                # New starts to reach target
                gap_to_target = accessible_patients - continuing
                if gap_to_target > 0:
                    # Need new patients
                    new_starts = gap_to_target * 0.3  # 30% of gap filled each year
                    
                    # Split between naive and switching
                    states['new_patients'][t] = new_starts * (1 - params.switching_in_rate)
                    states['switching_in'][t] = new_starts * params.switching_in_rate
                    
                    states['on_drug'][t] = continuing + states['new_patients'][t] + states['switching_in'][t]
                else:
                    # Above target, only discontinuation
                    states['on_drug'][t] = continuing
                    states['new_patients'][t] = 0
                    states['switching_in'][t] = 0
                
                # Track discontinued from previous year (before switching)
                discontinued_flow = states['on_drug'][t-1] * params.annual_discontinuation
                states['discontinued'][t] = states['discontinued'][t-1] + discontinued_flow
                
                # Switching out to competitors (separate from discontinuation)
                states['switching_out'][t] = states['on_drug'][t] * params.switching_out_rate
                states['on_drug'][t] -= states['switching_out'][t]
                
                # Switching out patients go to competitor drugs (mass conservation)
                states['competitor_drug'][t] = (states['competitor_drug'][t-1] if t > 0 else 0) + states['switching_out'][t]
        
        return states
    
    def calculate_revenue(self, patient_states: Dict[str, np.ndarray],
                         params: PatientFlowParams) -> np.ndarray:
        """
        Convert patient counts to revenue.
        
        Args:
            patient_states: Patient counts by state
            params: Flow parameters including pricing
        
        Returns:
            Revenue by year
        """
        
        # Treated patients × adherence × annual price × GTN
        revenue = (patient_states['on_drug'] * 
                  params.adherence_rate * 
                  params.annual_price * 
                  params.net_to_gross)
        
        return revenue
    
    def forecast(self, drug_row: pd.Series, years: int = 5) -> np.ndarray:
        """
        Generate revenue forecast using patient flow model.
        
        Args:
            drug_row: Drug characteristics
            years: Forecast horizon
        
        Returns:
            Revenue forecast
        """
        
        # Set parameters
        params = self.set_params_from_drug(drug_row)
        
        # Simulate patient flow
        patient_states = self.simulate_patient_flow(params, years)
        
        # Calculate revenue
        revenue = self.calculate_revenue(patient_states, params)
        
        return revenue
    
    def forecast_with_scenarios(self, drug_row: pd.Series, 
                              years: int = 5) -> Dict[str, np.ndarray]:
        """
        Generate base/upside/downside scenarios.
        
        Args:
            drug_row: Drug characteristics  
            years: Forecast horizon
        
        Returns:
            Dict with scenario forecasts
        """
        
        # Base case
        base_params = self.set_params_from_drug(drug_row)
        base_states = self.simulate_patient_flow(base_params, years)
        base_revenue = self.calculate_revenue(base_states, base_params)
        
        # Upside case
        upside_params = self.set_params_from_drug(drug_row)
        upside_params.peak_penetration *= 1.3
        upside_params.years_to_peak = max(2, upside_params.years_to_peak - 1)
        upside_params.adherence_rate = min(0.90, upside_params.adherence_rate * 1.2)
        upside_params.prior_auth_approval = min(0.95, upside_params.prior_auth_approval * 1.3)
        upside_states = self.simulate_patient_flow(upside_params, years)
        upside_revenue = self.calculate_revenue(upside_states, upside_params)
        
        # Downside case
        downside_params = self.set_params_from_drug(drug_row)
        downside_params.peak_penetration *= 0.6
        downside_params.years_to_peak += 2
        downside_params.adherence_rate *= 0.8
        downside_params.annual_discontinuation *= 1.5
        downside_params.prior_auth_approval *= 0.7
        downside_states = self.simulate_patient_flow(downside_params, years)
        downside_revenue = self.calculate_revenue(downside_states, downside_params)
        
        return {
            'base': base_revenue,
            'upside': upside_revenue,
            'downside': downside_revenue
        }
    
    def get_flow_diagnostics(self, drug_row: pd.Series, 
                            years: int = 5) -> Dict[str, any]:
        """
        Get detailed diagnostics of patient flow.
        
        Args:
            drug_row: Drug characteristics
            years: Forecast horizon
        
        Returns:
            Dict with flow metrics and diagnostics
        """
        
        params = self.set_params_from_drug(drug_row)
        states = self.simulate_patient_flow(params, years)
        revenue = self.calculate_revenue(states, params)
        
        # Calculate key metrics
        metrics = {
            'peak_patients': np.max(states['on_drug']),
            'peak_year': np.argmax(states['on_drug']),
            'peak_penetration_achieved': np.max(states['on_drug']) / np.max(states['eligible']),
            'cumulative_patients': np.sum(states['new_patients'] + states['switching_in']),
            'cumulative_discontinued': np.sum(states['on_drug'][:-1] * params.annual_discontinuation),
            'average_duration': 1 / params.annual_discontinuation if params.annual_discontinuation > 0 else np.inf,
            'peak_revenue': np.max(revenue),
            'cumulative_revenue': np.sum(revenue),
            'patient_states': states,
            'parameters': params
        }
        
        return metrics


def patient_flow_forecast(drug_row: pd.Series, years: int = 5) -> np.ndarray:
    """
    Convenience function for patient flow forecasting.
    
    Args:
        drug_row: Drug to forecast
        years: Forecast horizon
    
    Returns:
        Revenue forecast
    """
    model = PatientFlowModel()
    return model.forecast(drug_row, years)


def patient_flow_scenarios(drug_row: pd.Series, years: int = 5) -> Dict[str, np.ndarray]:
    """
    Generate scenario forecasts using patient flow.
    
    Args:
        drug_row: Drug to forecast
        years: Forecast horizon
    
    Returns:
        Dict with base/upside/downside scenarios
    """
    model = PatientFlowModel()
    return model.forecast_with_scenarios(drug_row, years)