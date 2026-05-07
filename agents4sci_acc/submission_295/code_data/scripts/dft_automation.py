#!/usr/bin/env python3
"""
DFT Calculation Automation Script
Automates density functional theory calculations for catalyst screening
Supports VASP, Quantum ESPRESSO, and GPAW
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read, write
from ase.build import bulk, fcc111, add_adsorbate
from ase.constraints import FixAtoms
from ase.optimize import BFGS
from ase.calculators.emt import EMT  # For testing
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
from dataclasses import dataclass, asdict


@dataclass
class DFTParameters:
    """DFT calculation parameters"""
    calculator: str = "vasp"  # vasp, qe, gpaw
    functional: str = "PBE"
    encut: float = 500  # eV for VASP
    kpoints: Tuple[int, int, int] = (4, 4, 1)
    spin_polarized: bool = True
    dipole_correction: bool = True
    vdw_correction: str = "D3"  # D3, TS, etc.
    convergence: Dict[str, float] = None
    
    def __post_init__(self):
        if self.convergence is None:
            self.convergence = {
                "energy": 1e-5,  # eV
                "forces": 0.02,  # eV/Å
                "stress": 0.1    # kbar
            }


@dataclass
class AdsorptionSite:
    """Adsorption site information"""
    name: str
    position: Tuple[float, float, float]
    site_type: str  # top, bridge, hollow, etc.


class DFTAutomation:
    def __init__(self, 
                 work_dir: str = "dft_calculations",
                 calculator: str = "vasp",
                 parallel_jobs: int = 4):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.calculator = calculator
        self.parallel_jobs = parallel_jobs
        self.results_db = []
        
    def setup_bulk_calculation(self,
                              formula: str,
                              structure_type: str = "fcc",
                              lattice_param: Optional[float] = None) -> Path:
        """Setup bulk structure calculation"""
        calc_dir = self.work_dir / f"bulk_{formula}"
        calc_dir.mkdir(parents=True, exist_ok=True)
        
        # Create bulk structure
        if structure_type == "fcc":
            atoms = bulk(formula, 'fcc', a=lattice_param or 4.0)
        elif structure_type == "bcc":
            atoms = bulk(formula, 'bcc', a=lattice_param or 3.0)
        elif structure_type == "hcp":
            atoms = bulk(formula, 'hcp', a=lattice_param or 3.0, c=lattice_param*1.6 if lattice_param else 4.8)
        else:
            raise ValueError(f"Unknown structure type: {structure_type}")
        
        # Write structure
        write(calc_dir / "POSCAR", atoms, format="vasp")
        
        # Write input files
        self._write_calculation_inputs(calc_dir, "bulk", DFTParameters())
        
        return calc_dir
    
    def setup_surface_calculation(self,
                                 bulk_atoms: Atoms,
                                 miller_indices: Tuple[int, int, int] = (1, 1, 1),
                                 layers: int = 4,
                                 vacuum: float = 15.0) -> Path:
        """Setup surface slab calculation"""
        formula = bulk_atoms.get_chemical_formula()
        calc_dir = self.work_dir / f"surface_{formula}_{miller_indices}"
        calc_dir.mkdir(parents=True, exist_ok=True)
        
        # Create surface slab
        if miller_indices == (1, 1, 1):
            slab = fcc111(formula.split()[0], size=(3, 3, layers), vacuum=vacuum)
        else:
            # For other surfaces, would use ase.build.surface
            from ase.build import surface
            slab = surface(bulk_atoms, miller_indices, layers, vacuum=vacuum)
        
        # Fix bottom layers
        positions = slab.get_positions()
        z_positions = positions[:, 2]
        bottom_atoms = z_positions < (z_positions.min() + 5.0)
        constraint = FixAtoms(indices=np.where(bottom_atoms)[0])
        slab.set_constraint(constraint)
        
        # Write structure
        write(calc_dir / "POSCAR", slab, format="vasp")
        
        # Write input files
        params = DFTParameters(kpoints=(3, 3, 1))  # Reduced k-points for slab
        self._write_calculation_inputs(calc_dir, "surface", params)
        
        return calc_dir
    
    def setup_adsorption_calculations(self,
                                    slab: Atoms,
                                    adsorbate: str,
                                    sites: Optional[List[AdsorptionSite]] = None) -> List[Path]:
        """Setup adsorption energy calculations"""
        if sites is None:
            sites = self._find_adsorption_sites(slab)
        
        calc_dirs = []
        formula = slab.get_chemical_formula()
        
        for site in sites:
            calc_dir = self.work_dir / f"ads_{formula}_{adsorbate}_{site.name}"
            calc_dir.mkdir(parents=True, exist_ok=True)
            
            # Create adsorbate molecule
            if adsorbate == "CO":
                ads_mol = Atoms('CO', [(0, 0, 0), (0, 0, 1.2)])
            elif adsorbate == "H":
                ads_mol = Atoms('H', [(0, 0, 0)])
            elif adsorbate == "OH":
                ads_mol = Atoms('OH', [(0, 0, 0), (0, 0, 0.97)])
            elif adsorbate == "CO2":
                ads_mol = Atoms('CO2', [(0, 0, 0), (-1.16, 0, 0), (1.16, 0, 0)])
            else:
                raise ValueError(f"Unknown adsorbate: {adsorbate}")
            
            # Add adsorbate to surface
            slab_ads = slab.copy()
            add_adsorbate(slab_ads, ads_mol, height=2.0, position=site.position[:2])
            
            # Write structure
            write(calc_dir / "POSCAR", slab_ads, format="vasp")
            
            # Write input files
            params = DFTParameters(kpoints=(3, 3, 1))
            self._write_calculation_inputs(calc_dir, "adsorption", params)
            
            calc_dirs.append(calc_dir)
        
        return calc_dirs
    
    def _find_adsorption_sites(self, slab: Atoms) -> List[AdsorptionSite]:
        """Find common adsorption sites on surface"""
        sites = []
        
        # Get top layer atoms
        positions = slab.get_positions()
        z_positions = positions[:, 2]
        top_layer = z_positions > (z_positions.max() - 1.0)
        top_indices = np.where(top_layer)[0]
        
        if len(top_indices) > 0:
            # Top site
            top_atom = top_indices[0]
            top_pos = positions[top_atom]
            sites.append(AdsorptionSite("top", tuple(top_pos), "top"))
            
            # Bridge site (between two atoms)
            if len(top_indices) > 1:
                bridge_pos = (positions[top_indices[0]] + positions[top_indices[1]]) / 2
                sites.append(AdsorptionSite("bridge", tuple(bridge_pos), "bridge"))
            
            # Hollow site (center of three atoms)
            if len(top_indices) > 2:
                hollow_pos = (positions[top_indices[0]] + 
                            positions[top_indices[1]] + 
                            positions[top_indices[2]]) / 3
                sites.append(AdsorptionSite("hollow", tuple(hollow_pos), "hollow"))
        
        return sites
    
    def _write_calculation_inputs(self, 
                                 calc_dir: Path, 
                                 calc_type: str,
                                 params: DFTParameters):
        """Write calculation input files"""
        if self.calculator == "vasp":
            self._write_vasp_inputs(calc_dir, calc_type, params)
        elif self.calculator == "qe":
            self._write_qe_inputs(calc_dir, calc_type, params)
        elif self.calculator == "gpaw":
            self._write_gpaw_inputs(calc_dir, calc_type, params)
        else:
            raise ValueError(f"Unknown calculator: {self.calculator}")
    
    def _write_vasp_inputs(self, calc_dir: Path, calc_type: str, params: DFTParameters):
        """Write VASP input files"""
        # INCAR
        incar = {
            "SYSTEM": f"{calc_type} calculation",
            "PREC": "Accurate",
            "ENCUT": params.encut,
            "ISMEAR": 1 if calc_type == "bulk" else 0,
            "SIGMA": 0.1,
            "EDIFF": params.convergence["energy"],
            "EDIFFG": -params.convergence["forces"],
            "ALGO": "Normal",
            "LREAL": "Auto",
            "LCHARG": ".FALSE.",
            "LWAVE": ".FALSE."
        }
        
        if params.spin_polarized:
            incar["ISPIN"] = 2
        
        if params.dipole_correction and calc_type in ["surface", "adsorption"]:
            incar.update({
                "LDIPOL": ".TRUE.",
                "IDIPOL": 3,
                "DIPOL": "0.5 0.5 0.5"
            })
        
        if params.vdw_correction:
            incar["IVDW"] = 11  # DFT-D3
        
        with open(calc_dir / "INCAR", 'w') as f:
            for key, value in incar.items():
                f.write(f"{key} = {value}\n")
        
        # KPOINTS
        with open(calc_dir / "KPOINTS", 'w') as f:
            f.write("K-points\n0\n")
            f.write("Monkhorst-Pack\n")
            f.write(f"{params.kpoints[0]} {params.kpoints[1]} {params.kpoints[2]}\n")
            f.write("0 0 0\n")
        
        # POTCAR - placeholder
        with open(calc_dir / "POTCAR_info.txt", 'w') as f:
            f.write("POTCAR files need to be copied from VASP pseudopotential library\n")
        
        # Job script
        self._write_job_script(calc_dir, "vasp")
    
    def _write_qe_inputs(self, calc_dir: Path, calc_type: str, params: DFTParameters):
        """Write Quantum ESPRESSO input files"""
        # Read structure
        atoms = read(calc_dir / "POSCAR")
        
        # Write QE input
        input_file = calc_dir / "pw.in"
        
        with open(input_file, 'w') as f:
            f.write("&CONTROL\n")
            f.write(f"  calculation = 'relax'\n")
            f.write(f"  prefix = '{calc_type}'\n")
            f.write(f"  pseudo_dir = './pseudo/'\n")
            f.write(f"  outdir = './tmp/'\n")
            f.write("/\n\n")
            
            f.write("&SYSTEM\n")
            f.write(f"  ibrav = 0\n")
            f.write(f"  nat = {len(atoms)}\n")
            f.write(f"  ntyp = {len(set(atoms.get_chemical_symbols()))}\n")
            f.write(f"  ecutwfc = {params.encut / 13.6}\n")  # Convert to Ry
            f.write(f"  ecutrho = {params.encut * 4 / 13.6}\n")
            if params.spin_polarized:
                f.write("  nspin = 2\n")
            f.write("/\n\n")
            
            f.write("&ELECTRONS\n")
            f.write(f"  conv_thr = {params.convergence['energy'] / 13.6}\n")
            f.write("/\n\n")
            
            f.write("&IONS\n")
            f.write("/\n\n")
        
        # Job script
        self._write_job_script(calc_dir, "qe")
    
    def _write_gpaw_inputs(self, calc_dir: Path, calc_type: str, params: DFTParameters):
        """Write GPAW Python script"""
        script = f"""#!/usr/bin/env python3
from ase.io import read, write
from gpaw import GPAW, PW
from ase.optimize import BFGS

# Read structure
atoms = read('POSCAR')

# Setup calculator
calc = GPAW(
    mode=PW({params.encut}),
    xc='{params.functional}',
    kpts={list(params.kpoints)},
    spinpol={params.spin_polarized},
    convergence={{
        'energy': {params.convergence['energy']},
        'forces': {params.convergence['forces']}
    }},
    txt='gpaw.txt'
)

atoms.set_calculator(calc)

# Optimize
opt = BFGS(atoms, trajectory='opt.traj')
opt.run(fmax={params.convergence['forces']})

# Calculate final energy
energy = atoms.get_potential_energy()
forces = atoms.get_forces()

# Save results
results = {{
    'energy': energy,
    'forces': forces.tolist(),
    'positions': atoms.get_positions().tolist()
}}

import json
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

write('CONTCAR', atoms)
"""
        
        with open(calc_dir / "run_gpaw.py", 'w') as f:
            f.write(script)
        
        # Job script
        self._write_job_script(calc_dir, "gpaw")
    
    def _write_job_script(self, calc_dir: Path, calculator: str):
        """Write job submission script"""
        script = f"""#!/bin/bash
#SBATCH --job-name={calc_dir.name}
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=24:00:00
#SBATCH --mem=32G

module load {calculator}

cd {calc_dir}

"""
        
        if calculator == "vasp":
            script += "mpirun vasp_std > vasp.out\n"
        elif calculator == "qe":
            script += "mpirun pw.x < pw.in > pw.out\n"
        elif calculator == "gpaw":
            script += "python run_gpaw.py > gpaw.out\n"
        
        with open(calc_dir / "job.sh", 'w') as f:
            f.write(script)
        
        # Make executable
        os.chmod(calc_dir / "job.sh", 0o755)
    
    def run_calculations(self, 
                        calc_dirs: List[Path],
                        submit: bool = False) -> List[Dict]:
        """Run or submit calculations"""
        results = []
        
        if submit:
            # Submit to queue
            for calc_dir in calc_dirs:
                try:
                    result = subprocess.run(
                        ["sbatch", "job.sh"],
                        cwd=calc_dir,
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        job_id = result.stdout.strip().split()[-1]
                        results.append({
                            "calc_dir": str(calc_dir),
                            "status": "submitted",
                            "job_id": job_id
                        })
                    else:
                        results.append({
                            "calc_dir": str(calc_dir),
                            "status": "failed",
                            "error": result.stderr
                        })
                        
                except Exception as e:
                    results.append({
                        "calc_dir": str(calc_dir),
                        "status": "error",
                        "error": str(e)
                    })
        else:
            # Run locally with EMT for testing
            print("Running test calculations with EMT calculator...")
            
            with ProcessPoolExecutor(max_workers=self.parallel_jobs) as executor:
                futures = {
                    executor.submit(self._run_emt_calculation, calc_dir): calc_dir
                    for calc_dir in calc_dirs
                }
                
                for future in as_completed(futures):
                    calc_dir = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        results.append({
                            "calc_dir": str(calc_dir),
                            "status": "error",
                            "error": str(e)
                        })
        
        return results
    
    def _run_emt_calculation(self, calc_dir: Path) -> Dict:
        """Run test calculation with EMT"""
        atoms = read(calc_dir / "POSCAR")
        atoms.set_calculator(EMT())
        
        # Optimize
        opt = BFGS(atoms, trajectory=calc_dir / "opt.traj")
        opt.run(fmax=0.05)
        
        # Get results
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        
        # Save results
        results = {
            "calc_dir": str(calc_dir),
            "status": "completed",
            "energy": energy,
            "forces": forces.tolist(),
            "max_force": float(np.max(np.abs(forces))),
            "positions": atoms.get_positions().tolist()
        }
        
        with open(calc_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        write(calc_dir / "CONTCAR", atoms)
        
        return results
    
    def calculate_adsorption_energies(self,
                                     surface_energy: float,
                                     adsorbate_results: List[Dict],
                                     adsorbate_reference: Dict[str, float]) -> Dict[str, float]:
        """Calculate adsorption energies from DFT results"""
        adsorption_energies = {}
        
        for result in adsorbate_results:
            if result["status"] != "completed":
                continue
            
            # Extract adsorbate and site from directory name
            calc_name = Path(result["calc_dir"]).name
            parts = calc_name.split("_")
            adsorbate = parts[-2]
            site = parts[-1]
            
            # E_ads = E_surf+ads - E_surf - E_ads_ref
            e_ads = result["energy"] - surface_energy - adsorbate_reference.get(adsorbate, 0)
            
            adsorption_energies[f"{adsorbate}_{site}"] = e_ads
        
        return adsorption_energies
    
    def generate_dft_report(self, 
                           catalyst: str,
                           results: Dict) -> Path:
        """Generate DFT calculation report"""
        report = f"""# DFT Calculation Report for {catalyst}
Generated: {datetime.now().isoformat()}

## Calculation Parameters
- Calculator: {self.calculator}
- Functional: PBE
- Spin-polarized: Yes

## Results Summary

### Bulk Properties
- Formation energy: {results.get('formation_energy', 'N/A')} eV/atom
- Lattice parameter: {results.get('lattice_param', 'N/A')} Å

### Surface Properties
- Surface energy: {results.get('surface_energy', 'N/A')} J/m²
- Work function: {results.get('work_function', 'N/A')} eV

### Adsorption Energies (eV)
"""
        
        if "adsorption_energies" in results:
            for ads_site, energy in results["adsorption_energies"].items():
                report += f"- {ads_site}: {energy:.3f} eV\n"
        
        report_file = self.work_dir / f"dft_report_{catalyst}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        return report_file


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automate DFT calculations")
    parser.add_argument("--catalyst", type=str, required=True,
                       help="Catalyst formula")
    parser.add_argument("--calculator", type=str, default="vasp",
                       choices=["vasp", "qe", "gpaw"],
                       help="DFT calculator to use")
    parser.add_argument("--adsorbates", nargs="+", default=["CO", "H"],
                       help="Adsorbates to calculate")
    parser.add_argument("--submit", action="store_true",
                       help="Submit to queue instead of running locally")
    
    args = parser.parse_args()
    
    # Initialize automation
    dft = DFTAutomation(calculator=args.calculator)
    
    # Setup calculations
    print(f"Setting up DFT calculations for {args.catalyst}")
    
    # 1. Bulk calculation
    bulk_dir = dft.setup_bulk_calculation(args.catalyst)
    print(f"Bulk calculation: {bulk_dir}")
    
    # 2. Surface calculation
    bulk_atoms = bulk(args.catalyst, 'fcc', a=4.0)  # Simplified
    surf_dir = dft.setup_surface_calculation(bulk_atoms)
    print(f"Surface calculation: {surf_dir}")
    
    # 3. Adsorption calculations
    slab = read(surf_dir / "POSCAR")
    ads_dirs = []
    for adsorbate in args.adsorbates:
        dirs = dft.setup_adsorption_calculations(slab, adsorbate)
        ads_dirs.extend(dirs)
        print(f"Adsorption calculations for {adsorbate}: {len(dirs)} sites")
    
    # Run calculations
    all_dirs = [bulk_dir, surf_dir] + ads_dirs
    results = dft.run_calculations(all_dirs, submit=args.submit)
    
    # Summary
    print(f"\nCalculation summary:")
    for result in results:
        print(f"- {Path(result['calc_dir']).name}: {result['status']}")


if __name__ == "__main__":
    main()