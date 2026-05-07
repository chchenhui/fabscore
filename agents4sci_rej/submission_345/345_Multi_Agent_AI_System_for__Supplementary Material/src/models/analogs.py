"""
Analog-based forecasting using historical similar drugs.
This is what real consultants use - find similar historical launches.
Following Linus principle: Real data beats theory every time.

Enhanced per GPT-5 guidance (Day 3):
- TA priors integration for missing data
- Similarity by year-since-launch correlation/DTW
- Normalize curves by Y2 or peak
- Cap analog weight ≤0.5, same TA required
- Ensemble fallback if inputs missing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from scipy.stats import pearsonr
import warnings
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import TA priors for enhanced functionality
try:
    from .ta_priors import (
        apply_ta_priors, get_ta_analog_weights, get_ta_peak_share_prior,
        get_ta_access_tier_multiplier, get_ta_growth_rate, get_ta_factor_caps,
        get_ta_prediction_interval_multipliers, get_ta_minimum_revenue_threshold,
        get_ta_y2_share_prior
    )
except ImportError:
    # Fallback if ta_priors not available
    apply_ta_priors = None
    get_ta_analog_weights = None
    get_ta_peak_share_prior = None
    get_ta_access_tier_multiplier = None
    get_ta_growth_rate = None
    get_ta_factor_caps = None
    get_ta_prediction_interval_multipliers = None
    get_ta_minimum_revenue_threshold = None
    get_ta_y2_share_prior = None


def choose_normalization_base(
    trajectory: np.ndarray, 
    method: str, 
    therapeutic_area: str
) -> Optional[Tuple[float, str]]:
    """
    Choose consistent normalization base for trajectory scaling.
    
    Args:
        trajectory: Revenue trajectory array
        method: 'y2' or 'peak'
        therapeutic_area: TA for getting priors
    
    Returns:
        Tuple of (base_value, base_type) or None if invalid
    """
    
    if method == 'y2':
        # Prefer Y2 if present
        if len(trajectory) > 1 and trajectory[1] > 0:
            return trajectory[1], 'y2_actual'
        
        # Else median of Y0-Y2 only (true early years for Y2 estimation)
        early_years = trajectory[:3]  # Y0-Y2 only
        nonzero_early = early_years[early_years > 0]
        if len(nonzero_early) > 0:
            return np.median(nonzero_early), 'early_median'
        
        # Else use peak × ta_prior_y2_share
        peak_revenue = np.max(trajectory) if np.max(trajectory) > 0 else 0
        if peak_revenue > 0 and get_ta_y2_share_prior is not None:
            y2_share = get_ta_y2_share_prior(therapeutic_area)
            peak_share = get_ta_peak_share_prior(therapeutic_area) if get_ta_peak_share_prior else 0.15
            estimated_y2 = peak_revenue * (y2_share / peak_share)
            return estimated_y2, 'peak_derived'
            
    elif method == 'peak':
        # Use peak revenue
        peak_revenue = np.max(trajectory) if np.max(trajectory) > 0 else 0
        if peak_revenue > 0:
            return peak_revenue, 'peak_actual'
    
    return None


def dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """
    Dynamic Time Warping distance between two sequences.
    Used for comparing revenue trajectories of different lengths.
    """
    n, m = len(seq1), len(seq2)
    
    # Initialize DTW matrix
    dtw = np.zeros((n + 1, m + 1))
    dtw[0, 1:] = np.inf
    dtw[1:, 0] = np.inf
    
    # Fill DTW matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(seq1[i-1] - seq2[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    
    return dtw[n, m]


class AnalogForecaster:
    """
    Forecast based on historical analogs.
    Industry standard approach used by LEK, ZS, IQVIA.
    
    Enhanced with:
    - TA priors for missing data
    - Trajectory similarity metrics (correlation/DTW)
    - Y2/peak normalization
    - Capped analog weights
    """
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path(__file__).parent.parent.parent / "data_proc"
        self.launches = None
        self.revenues = None
        self.analogs = None
        self.ta_imputations = None
        self._load_data()
    
    def _load_data(self):
        """Load processed datasets including TA-imputed data."""
        try:
            self.launches = pd.read_parquet(self.data_dir / "launches.parquet")
            self.revenues = pd.read_parquet(self.data_dir / "launch_revenues.parquet")
            self.analogs = pd.read_parquet(self.data_dir / "analogs.parquet")
            
            # Load TA-imputed data if available
            ta_impute_path = self.data_dir / "ta_imputations.parquet"
            if ta_impute_path.exists():
                self.ta_imputations = pd.read_parquet(ta_impute_path)
                
        except FileNotFoundError:
            # Allow initialization without data for testing
            pass
    
    def find_best_analogs_enhanced(
        self, 
        drug_row: pd.Series,
        n_analogs: int = 5,
        require_same_ta: bool = True
    ) -> List[str]:
        """
        Enhanced analog finding with TA constraints and similarity scoring.
        
        Args:
            drug_row: Target drug
            n_analogs: Number of analogs to find
            require_same_ta: Only consider drugs in same TA
            
        Returns:
            List of launch_ids for best analogs
        """
        
        if self.launches is None or self.revenues is None:
            return []
        
        # Get candidate analogs
        candidates = self.launches.copy()
        
        # Exclude the drug itself
        candidates = candidates[candidates['launch_id'] != drug_row['launch_id']]
        
        # Filter by TA if required
        if require_same_ta:
            drug_ta = drug_row.get('therapeutic_area', 'Unknown')
            candidates = candidates[candidates['therapeutic_area'] == drug_ta]
            
            # If no same-TA drugs, fall back to weighted cross-TA
            if len(candidates) == 0 and get_ta_analog_weights is not None:
                candidates = self.launches[self.launches['launch_id'] != drug_row['launch_id']]
                require_same_ta = False
        
        # Calculate enhanced similarities
        similarities = []
        
        for _, candidate in candidates.iterrows():
            # Get revenue trajectory for similarity calculation
            candidate_revs = self.revenues[
                self.revenues['launch_id'] == candidate['launch_id']
            ].sort_values('year_since_launch')
            
            if len(candidate_revs) < 2:
                continue
                
            candidate_traj = candidate_revs['revenue_usd'].values
            
            # Calculate characteristic similarity
            char_sim = self._calculate_characteristic_similarity(drug_row, candidate)
            
            # Apply TA weight if cross-TA
            ta_weight = 1.0
            if not require_same_ta and get_ta_analog_weights is not None:
                drug_ta = drug_row.get('therapeutic_area', 'Unknown')
                ta_weights = get_ta_analog_weights(drug_ta)
                candidate_ta = candidate.get('therapeutic_area', 'Unknown')
                ta_weight = ta_weights.get(candidate_ta, 0.1)
            
            # Combined similarity with cap at 0.5
            total_sim = min(char_sim * ta_weight, 0.5)
            
            similarities.append((candidate['launch_id'], total_sim))
        
        # Sort and return top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [aid for aid, _ in similarities[:n_analogs]]
    
    def _calculate_characteristic_similarity(
        self,
        drug1: pd.Series,
        drug2: pd.Series
    ) -> float:
        """
        Calculate similarity based on drug characteristics.
        """
        
        similarities = []
        
        # Market size similarity
        if drug1.get('eligible_patients_at_launch', 0) > 0 and drug2.get('eligible_patients_at_launch', 0) > 0:
            size_ratio = min(
                drug1['eligible_patients_at_launch'] / drug2['eligible_patients_at_launch'],
                drug2['eligible_patients_at_launch'] / drug1['eligible_patients_at_launch']
            )
            similarities.append(size_ratio)
        
        # Price similarity
        if drug1.get('list_price_month_usd_launch', 0) > 0 and drug2.get('list_price_month_usd_launch', 0) > 0:
            price_ratio = min(
                drug1['list_price_month_usd_launch'] / drug2['list_price_month_usd_launch'],
                drug2['list_price_month_usd_launch'] / drug1['list_price_month_usd_launch']
            )
            similarities.append(price_ratio)
        
        # GTN similarity
        if drug1.get('net_gtn_pct_launch', 0) > 0 and drug2.get('net_gtn_pct_launch', 0) > 0:
            gtn_diff = abs(drug1['net_gtn_pct_launch'] - drug2['net_gtn_pct_launch'])
            gtn_sim = 1 - gtn_diff
            similarities.append(gtn_sim)
        
        # Access tier similarity using TA-based multipliers
        drug_ta = drug1.get('therapeutic_area', drug2.get('therapeutic_area', 'Immunology'))
        access_map = get_ta_access_tier_multiplier(drug_ta) if get_ta_access_tier_multiplier else {'OPEN': 1.0, 'PA': 0.65, 'NICHE': 0.35}
        tier1 = access_map.get(drug1.get('access_tier_at_launch', 'PA'), 0.6)
        tier2 = access_map.get(drug2.get('access_tier_at_launch', 'PA'), 0.6)
        access_sim = 1 - abs(tier1 - tier2)
        similarities.append(access_sim)
        
        # Return average similarity
        if similarities:
            return np.mean(similarities)
        return 0.5
    
    def find_best_analogs(self, drug_row: pd.Series, n_analogs: int = 5) -> List[str]:
        """
        Find the best historical analogs for a drug.
        
        Args:
            drug_row: Row from launches.parquet
            n_analogs: Number of analogs to use
        
        Returns:
            List of launch_ids for best analogs
        """
        
        if self.analogs is None:
            return []
        
        # Get analogs for this drug
        drug_analogs = self.analogs[
            self.analogs['launch_id'] == drug_row['launch_id']
        ].sort_values('similarity_score', ascending=False)
        
        # Take top N
        best_analogs = drug_analogs.head(n_analogs)['analog_launch_id'].tolist()
        
        return best_analogs
    
    def get_analog_trajectories(self, analog_ids: List[str]) -> pd.DataFrame:
        """
        Get revenue trajectories for analog drugs.
        
        Args:
            analog_ids: List of launch_ids
        
        Returns:
            DataFrame with analog revenue trajectories
        """
        
        if self.revenues is None:
            return pd.DataFrame()
        
        # Get revenue data for analogs
        analog_revenues = self.revenues[
            self.revenues['launch_id'].isin(analog_ids)
        ]
        
        return analog_revenues
    
    def normalize_trajectories(self, drug_row: pd.Series, 
                              analog_revenues: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize analog trajectories to target drug characteristics.
        
        Args:
            drug_row: Target drug characteristics
            analog_revenues: Historical analog revenues
        
        Returns:
            Normalized revenue trajectories
        """
        
        if self.launches is None or analog_revenues.empty:
            return pd.DataFrame()
        
        normalized = []
        
        for analog_id in analog_revenues['launch_id'].unique():
            # Get analog characteristics
            analog_row = self.launches[
                self.launches['launch_id'] == analog_id
            ].iloc[0]
            
            # Get analog trajectory
            analog_traj = analog_revenues[
                analog_revenues['launch_id'] == analog_id
            ].copy()
            
            # Normalize for market size (with fallback)
            market_ratio = 1.0
            if (drug_row.get('eligible_patients_at_launch', 0) > 0 and 
                analog_row.get('eligible_patients_at_launch', 0) > 0):
                market_ratio = (drug_row['eligible_patients_at_launch'] / 
                              analog_row['eligible_patients_at_launch'])
            
            # Normalize for price (with fallback)
            price_ratio = 1.0
            if (drug_row.get('list_price_month_usd_launch', 0) > 0 and 
                analog_row.get('list_price_month_usd_launch', 0) > 0):
                price_ratio = ((drug_row['list_price_month_usd_launch'] * 12) /
                             (analog_row['list_price_month_usd_launch'] * 12))
            
            # Normalize for GTN (with fallback)
            gtn_ratio = 1.0
            if (drug_row.get('net_gtn_pct_launch', 0) > 0 and 
                analog_row.get('net_gtn_pct_launch', 0) > 0):
                gtn_ratio = (drug_row['net_gtn_pct_launch'] / 
                           analog_row['net_gtn_pct_launch'])
            
            # Apply normalization
            analog_traj['normalized_revenue'] = (
                analog_traj['revenue_usd'] * 
                market_ratio * price_ratio * gtn_ratio
            )
            
            # Adjust for access tier differences
            therapeutic_area = drug_row.get('therapeutic_area', analog_row.get('therapeutic_area', 'Immunology'))
            access_mult = self._get_access_adjustment(
                drug_row.get('access_tier_at_launch', 'PA'),
                analog_row.get('access_tier_at_launch', 'PA'),
                therapeutic_area
            )
            analog_traj['normalized_revenue'] *= access_mult
            
            # Adjust for efficacy differences (with fallback)
            efficacy_mult = 1.0
            if (drug_row.get('clinical_efficacy_proxy', 0) > 0 and 
                analog_row.get('clinical_efficacy_proxy', 0) > 0):
                efficacy_mult = (drug_row['clinical_efficacy_proxy'] / 
                               analog_row['clinical_efficacy_proxy'])
            analog_traj['normalized_revenue'] *= efficacy_mult
            
            normalized.append(analog_traj)
        
        if normalized:
            return pd.concat(normalized)
        return pd.DataFrame()
    
    def _get_access_adjustment(self, target_tier: str, analog_tier: str, therapeutic_area: str = 'Immunology') -> float:
        """
        Get adjustment factor for access tier differences using TA-based multipliers.
        
        Args:
            target_tier: Target drug access tier
            analog_tier: Analog drug access tier
            therapeutic_area: TA for getting multipliers
        
        Returns:
            Adjustment multiplier
        """
        
        # Use TA-based access tier values
        tier_values = get_ta_access_tier_multiplier(therapeutic_area) if get_ta_access_tier_multiplier else {
            'OPEN': 1.0, 'PA': 0.65, 'NICHE': 0.35
        }
        
        target_value = tier_values.get(target_tier, 0.6)
        analog_value = tier_values.get(analog_tier, 0.6)
        
        return target_value / analog_value if analog_value > 0 else 1.0
    
    def forecast_from_analogs(self, drug_row: pd.Series, 
                             years: int = 5,
                             n_analogs: int = 5,
                             method: str = 'weighted_mean',
                             normalization: str = 'y2',
                             require_same_ta: bool = True) -> np.ndarray:
        """
        Generate forecast using analog method with enhancements.
        
        Args:
            drug_row: Target drug to forecast
            years: Number of years to forecast
            n_analogs: Number of analogs to use
            method: Aggregation method ('mean', 'median', 'weighted_mean')
            normalization: 'y2' or 'peak' normalization (new)
            require_same_ta: Only use same TA drugs (new)
        
        Returns:
            Revenue forecast array
        """
        
        # Apply TA priors if needed
        drug_row, _ = self.apply_ta_priors_if_needed(drug_row)
        
        # Find best analogs with enhanced logic
        analog_ids = self.find_best_analogs_enhanced(drug_row, n_analogs, require_same_ta)
        
        if not analog_ids:
            # No analogs found, use simple growth
            return self._fallback_forecast(drug_row, years)
        
        # Get analog trajectories
        analog_revenues = self.get_analog_trajectories(analog_ids)
        
        if analog_revenues.empty:
            return self._fallback_forecast(drug_row, years)
        
        # Normalize trajectories with enhanced method
        if normalization in ['y2', 'peak']:
            # Use proper normalization from real analog data
            forecast_curves = []
            weights = []
            
            for analog_id in analog_ids:
                # Get analog's actual revenue trajectory from real data
                analog_traj = analog_revenues[
                    analog_revenues['launch_id'] == analog_id
                ].sort_values('year_since_launch')
                
                # Filter: require ≥3 revenue years for stable normalization
                if len(analog_traj) < 3:
                    continue
                
                # Get analog drug characteristics from real data
                analog_row = self.launches[self.launches['launch_id'] == analog_id]
                if analog_row.empty:
                    continue
                analog_row = analog_row.iloc[0]
                
                # Build analog's actual trajectory
                traj = np.zeros(years)
                for _, row in analog_traj.iterrows():
                    if row['year_since_launch'] < years:
                        traj[int(row['year_since_launch'])] = row['revenue_usd']
                
                # Skip if trajectory is empty
                if np.sum(traj) == 0:
                    continue
                
                # Get normalization base for this analog
                base_result = choose_normalization_base(traj, normalization, analog_row['therapeutic_area'])
                if not base_result:
                    continue
                base_revenue, base_type = base_result
                
                # Normalize using the SAME base as we'll use for rescaling (critical fix)
                if base_revenue > 0:
                    norm_traj = traj / base_revenue
                else:
                    print(f"[ANALOG] ERROR: base_revenue is zero for {analog_id}")
                    continue
                
                # Calculate scaling factors based on real drug characteristics
                # Market size adjustment
                size_factor = 1.0
                if analog_row.get('eligible_patients_at_launch', 0) > 0 and drug_row.get('eligible_patients_at_launch', 0) > 0:
                    size_factor = drug_row['eligible_patients_at_launch'] / analog_row['eligible_patients_at_launch']
                
                # Price adjustment (annual)
                price_factor = 1.0
                if analog_row.get('list_price_month_usd_launch', 0) > 0 and drug_row.get('list_price_month_usd_launch', 0) > 0:
                    price_factor = (drug_row['list_price_month_usd_launch'] * 12) / (analog_row['list_price_month_usd_launch'] * 12)
                
                # GTN adjustment
                gtn_factor = 1.0
                if analog_row.get('net_gtn_pct_launch', 0) > 0 and drug_row.get('net_gtn_pct_launch', 0) > 0:
                    gtn_factor = drug_row['net_gtn_pct_launch'] / analog_row['net_gtn_pct_launch']
                
                # Access tier adjustment
                access_factor = self._get_access_adjustment(
                    drug_row.get('access_tier_at_launch', 'PA'),
                    analog_row.get('access_tier_at_launch', 'PA')
                )
                
                # Efficacy adjustment
                efficacy_factor = 1.0
                if analog_row.get('clinical_efficacy_proxy', 0) > 0 and drug_row.get('clinical_efficacy_proxy', 0) > 0:
                    efficacy_factor = drug_row['clinical_efficacy_proxy'] / analog_row['clinical_efficacy_proxy']
                
                # Use consistent normalization base (already calculated above)
                analog_ta = analog_row.get('therapeutic_area', drug_row.get('therapeutic_area', 'Immunology'))
                
                # Apply minimum threshold check
                min_threshold = get_ta_minimum_revenue_threshold(analog_ta) if get_ta_minimum_revenue_threshold else 5e6
                if base_revenue < min_threshold:
                    print(f"[ANALOG] Skipping {analog_id}: base_revenue ${base_revenue/1e6:.1f}M < threshold ${min_threshold/1e6:.1f}M")
                    continue
                
                # Apply TA-based factor caps (no hardcoded limits)
                drug_ta = drug_row.get('therapeutic_area', 'Immunology')
                factor_caps = get_ta_factor_caps(drug_ta) if get_ta_factor_caps else {
                    'size_factor': (0.1, 10.0), 'price_factor': (0.2, 5.0), 'gtn_factor': (0.4, 2.0),
                    'access_factor': (0.3, 3.0), 'efficacy_factor': (0.5, 2.0)
                }
                
                # Apply caps with fallback to 1.0 if zero
                size_min, size_max = factor_caps['size_factor']
                size_factor = max(min(size_factor, size_max), size_min) if size_factor > 0 else 1.0
                
                price_min, price_max = factor_caps['price_factor']
                price_factor = max(min(price_factor, price_max), price_min) if price_factor > 0 else 1.0
                
                gtn_min, gtn_max = factor_caps['gtn_factor']
                gtn_factor = max(min(gtn_factor, gtn_max), gtn_min) if gtn_factor > 0 else 1.0
                
                access_min, access_max = factor_caps['access_factor']
                access_factor = max(min(access_factor, access_max), access_min) if access_factor > 0 else 1.0
                
                efficacy_min, efficacy_max = factor_caps['efficacy_factor']
                efficacy_factor = max(min(efficacy_factor, efficacy_max), efficacy_min) if efficacy_factor > 0 else 1.0
                
                adjusted_revenue = base_revenue * size_factor * price_factor * gtn_factor * access_factor * efficacy_factor
                
                # Apply normalized shape with adjusted scale
                forecast_curve = norm_traj * adjusted_revenue
                
                # Apply market-based caps (no absolute limits)
                patients = drug_row.get('eligible_patients_at_launch', 0)
                annual_price = drug_row.get('list_price_month_usd_launch', 0) * 12
                gtn = drug_row.get('net_gtn_pct_launch', 0)
                access_tier = drug_row.get('access_tier_at_launch', 'PA')
                
                if patients > 0 and annual_price > 0 and gtn > 0:
                    # Get TA-specific peak share ceiling
                    peak_share = get_ta_peak_share_prior(drug_ta) if get_ta_peak_share_prior else 0.15
                    
                    # Get access tier multiplier
                    access_multipliers = get_ta_access_tier_multiplier(drug_ta) if get_ta_access_tier_multiplier else {'OPEN': 1.0, 'PA': 0.65, 'NICHE': 0.35}
                    access_mult = access_multipliers.get(access_tier, 0.65)
                    
                    # Calculate market capacity
                    market_capacity = patients * annual_price * gtn * peak_share * access_mult
                    
                    # Cap each year by market capacity
                    forecast_curve = np.minimum(forecast_curve, market_capacity)
                else:
                    print(f"[ANALOG] Warning: Missing market data for {drug_row.get('drug_name', 'unknown')}, no market cap applied")
                
                # Cap individual analog contribution (per GPT-5 guidance)
                if self.analogs is not None:
                    analog_sim = self.analogs[
                        (self.analogs['launch_id'] == drug_row['launch_id']) &
                        (self.analogs['analog_launch_id'] == analog_id)
                    ]['similarity_score'].values
                    weight = min(analog_sim[0], 0.5) if len(analog_sim) > 0 else 0.5
                else:
                    weight = 0.5 / len(analog_ids)  # Equal weights capped at 0.5 total
                
                forecast_curves.append(forecast_curve)
                weights.append(weight)
            
            if len(forecast_curves) >= 2:  # Need at least 2 analogs for stable averaging
                # Implement proper weight cap after normalization
                forecast_curves = np.array(forecast_curves)
                weights = np.array(weights)
                
                # Normalize weights to sum to 1
                weights = weights / weights.sum()
                
                # Apply 0.5 cap constraint per GPT-5 guidance
                max_weight_exceeded = np.any(weights > 0.5)
                if max_weight_exceeded:
                    # Cap weights at 0.5 and redistribute
                    capped_weights = np.minimum(weights, 0.5)
                    remaining_mass = 1.0 - np.sum(capped_weights)
                    uncapped_indices = weights <= 0.5
                    
                    if np.any(uncapped_indices) and remaining_mass > 0:
                        # Redistribute remaining mass to uncapped weights
                        remaining_weights = weights[uncapped_indices]
                        if np.sum(remaining_weights) > 0:
                            scale_factor = remaining_mass / np.sum(remaining_weights)
                            capped_weights[uncapped_indices] = remaining_weights * scale_factor
                    
                    weights = capped_weights
                
                forecast = np.average(forecast_curves, weights=weights, axis=0)
            else:
                # Fall back to ensemble if no valid analogs  
                print(f"[ANALOG] No valid analogs found (need >=3 revenue years), using ensemble fallback")
                forecast = self._fallback_forecast(drug_row, years)
        else:
            # Original normalization method
            normalized = self.normalize_trajectories(drug_row, analog_revenues)
            
            # Aggregate by year
            forecast = np.zeros(years)
            
            for year in range(years):
                year_data = normalized[
                    normalized['year_since_launch'] == year
                ]['normalized_revenue']
            
            if len(year_data) > 0:
                if method == 'mean':
                    forecast[year] = year_data.mean()
                elif method == 'median':
                    forecast[year] = year_data.median()
                elif method == 'weighted_mean':
                    # Weight by similarity score - properly align values with analog IDs
                    weights = []
                    values = []
                    
                    # Get normalized revenues for this year grouped by analog
                    year_normalized = normalized[
                        normalized['year_since_launch'] == year
                    ]
                    
                    for analog_id in analog_ids:
                        # Get revenue for this analog at this year
                        analog_revenue = year_normalized[
                            year_normalized['launch_id'] == analog_id
                        ]['normalized_revenue'].values
                        
                        if len(analog_revenue) > 0:
                            # Get similarity score for this analog
                            sim_score = self.analogs[
                                (self.analogs['launch_id'] == drug_row['launch_id']) &
                                (self.analogs['analog_launch_id'] == analog_id)
                            ]['similarity_score'].values
                            
                            if len(sim_score) > 0:
                                weights.append(sim_score[0])
                                values.append(analog_revenue[0])
                    
                    if weights:
                        forecast[year] = np.average(values, weights=weights)
                    else:
                        forecast[year] = year_data.mean()
            else:
                # Extrapolate if no data using TA-specific growth
                if year > 0 and forecast[year-1] > 0:
                    drug_ta = drug_row.get('therapeutic_area', 'Immunology')
                    growth_rate = get_ta_growth_rate(drug_ta) if get_ta_growth_rate else 1.15
                    forecast[year] = forecast[year-1] * growth_rate
        
        return forecast
    
    def _fallback_forecast(self, drug_row: pd.Series, years: int) -> np.ndarray:
        """Simple fallback when no analogs available."""
        
        # Import baseline method as fallback
        from .baselines import linear_trend_forecast
        return linear_trend_forecast(drug_row, years)
    
    def apply_ta_priors_if_needed(self, drug_row: pd.Series) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Apply TA priors to fill missing data fields.
        
        Returns:
            Tuple of (updated drug_row, imputation_log)
        """
        imputation_log = {}
        
        # Skip if TA priors not available
        if apply_ta_priors is None:
            return drug_row, {"error": "TA priors not available"}
        
        # Check if we have pre-computed imputations
        if self.ta_imputations is not None:
            drug_imputed = self.ta_imputations[
                self.ta_imputations['launch_id'] == drug_row['launch_id']
            ]
            if not drug_imputed.empty:
                return drug_imputed.iloc[0], {"source": "pre-computed"}
        
        # Otherwise apply priors on the fly
        drug_row = apply_ta_priors(drug_row, imputation_log)
        return drug_row, imputation_log
    
    def calculate_trajectory_similarity(
        self, 
        traj1: np.ndarray, 
        traj2: np.ndarray,
        method: str = 'correlation'
    ) -> float:
        """
        Calculate similarity between two revenue trajectories.
        
        Args:
            traj1: First trajectory
            traj2: Second trajectory  
            method: 'correlation' or 'dtw'
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        
        if len(traj1) < 2 or len(traj2) < 2:
            return 0.0
            
        # Align lengths by taking minimum
        min_len = min(len(traj1), len(traj2))
        traj1 = traj1[:min_len]
        traj2 = traj2[:min_len]
        
        if method == 'correlation':
            # Pearson correlation
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr, _ = pearsonr(traj1, traj2)
                # Convert to 0-1 range
                return max(0, (corr + 1) / 2)
                
        elif method == 'dtw':
            # Dynamic time warping
            # Normalize by peak for DTW
            peak1 = np.max(traj1) if np.max(traj1) > 0 else 1
            peak2 = np.max(traj2) if np.max(traj2) > 0 else 1
            norm_traj1 = traj1 / peak1
            norm_traj2 = traj2 / peak2
            
            distance = dtw_distance(norm_traj1, norm_traj2)
            # Convert distance to similarity (inverse)
            # Normalize by trajectory length
            max_dist = len(norm_traj1) * 2  # Maximum possible distance
            similarity = 1 - min(distance / max_dist, 1)
            return similarity
            
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def normalize_trajectory(
        self,
        trajectory: np.ndarray,
        method: str = 'y2'
    ) -> np.ndarray:
        """
        Normalize revenue trajectory by Y2 or peak.
        
        Args:
            trajectory: Revenue trajectory
            method: 'y2' or 'peak'
            
        Returns:
            Normalized trajectory
        """
        
        if len(trajectory) < 2:
            return trajectory
            
        if method == 'y2':
            # Normalize by year 2 revenue
            if len(trajectory) > 1 and trajectory[1] > 0:
                return trajectory / trajectory[1]
            elif trajectory[0] > 0:
                return trajectory / trajectory[0]
                
        elif method == 'peak':
            # Normalize by peak revenue
            peak = np.max(trajectory)
            if peak > 0:
                return trajectory / peak
                
        return trajectory
    
    def get_prediction_intervals(self, drug_row: pd.Series,
                                years: int = 5,
                                n_analogs: int = 10,
                                confidence: float = 0.8) -> Dict[str, np.ndarray]:
        """
        Generate prediction intervals using analog variability.
        
        Args:
            drug_row: Target drug
            years: Forecast horizon
            n_analogs: Number of analogs for interval estimation
            confidence: Confidence level (0.8 = 80% PI)
        
        Returns:
            Dict with 'forecast', 'lower', 'upper' arrays
        """
        
        # Get more analogs for interval estimation
        analog_ids = self.find_best_analogs(drug_row, n_analogs)
        
        if not analog_ids:
            # No intervals without analogs - use TA-based multipliers
            forecast = self._fallback_forecast(drug_row, years)
            drug_ta = drug_row.get('therapeutic_area', 'Immunology')
            lower_mult, upper_mult = get_ta_prediction_interval_multipliers(drug_ta) if get_ta_prediction_interval_multipliers else (0.7, 1.3)
            return {
                'forecast': forecast,
                'lower': forecast * lower_mult,
                'upper': forecast * upper_mult
            }
        
        # Get all analog trajectories
        analog_revenues = self.get_analog_trajectories(analog_ids)
        normalized = self.normalize_trajectories(drug_row, analog_revenues)
        
        # Calculate percentiles by year
        forecast = np.zeros(years)
        lower = np.zeros(years)
        upper = np.zeros(years)
        
        alpha = (1 - confidence) / 2
        
        for year in range(years):
            year_data = normalized[
                normalized['year_since_launch'] == year
            ]['normalized_revenue'].values
            
            if len(year_data) > 2:
                forecast[year] = np.median(year_data)
                lower[year] = np.percentile(year_data, alpha * 100)
                upper[year] = np.percentile(year_data, (1-alpha) * 100)
            elif len(year_data) > 0:
                forecast[year] = np.mean(year_data)
                # Wide intervals with limited data - use TA multipliers
                drug_ta = drug_row.get('therapeutic_area', 'Immunology')
                lower_mult, upper_mult = get_ta_prediction_interval_multipliers(drug_ta) if get_ta_prediction_interval_multipliers else (0.5, 1.5)
                lower[year] = forecast[year] * lower_mult
                upper[year] = forecast[year] * upper_mult
            else:
                # Extrapolate using TA growth rate
                if year > 0:
                    drug_ta = drug_row.get('therapeutic_area', 'Immunology')
                    growth = get_ta_growth_rate(drug_ta) if get_ta_growth_rate else 1.15
                    forecast[year] = forecast[year-1] * growth
                    lower[year] = lower[year-1] * growth
                    upper[year] = upper[year-1] * growth
        
        return {
            'forecast': forecast,
            'lower': lower,
            'upper': upper
        }


def analog_forecast(drug_row: pd.Series, years: int = 5) -> np.ndarray:
    """
    Convenience function for analog forecasting.
    
    Args:
        drug_row: Drug to forecast
        years: Forecast horizon
    
    Returns:
        Revenue forecast
    """
    forecaster = AnalogForecaster()
    return forecaster.forecast_from_analogs(drug_row, years)


def analog_forecast_with_pi(drug_row: pd.Series, years: int = 5) -> Dict[str, np.ndarray]:
    """
    Analog forecast with prediction intervals.
    
    Args:
        drug_row: Drug to forecast
        years: Forecast horizon
    
    Returns:
        Dict with forecast and intervals
    """
    forecaster = AnalogForecaster()
    return forecaster.get_prediction_intervals(drug_row, years)