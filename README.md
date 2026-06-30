# GJF → DFT (B3LYP/LANL2DZ) Optimization Pipeline

A Google Colab pipeline that takes a Gaussian input file (`.gjf`) and runs a full DFT geometry optimization using [PySCF](https://pyscf.org/) and [geomeTRIC](https://github.com/leeping/geomeTRIC) — including support for transition metals like Fe and Zn via LANL2DZ effective core potentials (ECPs).

## What it does

1. Parses atoms, charge, and multiplicity from a `.gjf` file
2. Builds a PySCF molecule with per-element LANL2DZ basis + ECP assignment
3. Runs a B3LYP geometry optimization via geomeTRIC (automatically switches to unrestricted DFT for open-shell/high-spin systems based on parsed multiplicity)
4. Renders the optimized structure in 3D inline (via py3Dmol)
5. Packages results into a downloadable ZIP:
   - `optimized.xyz`
   - `optimized.gjf`
   - `optimization.log`

## Why B3LYP/LANL2DZ

B3LYP is a well-validated, general-purpose hybrid functional. LANL2DZ pairs a double-zeta basis for light elements with effective core potentials for heavier atoms (transition metals, etc.), making it a practical default when your molecule includes metals like Fe or Zn without needing a much heavier all-electron basis set.

This is a step up in accuracy (and computational cost) from a fast GFN2-xTB pre-optimization — use xtb first for a quick, reasonable starting geometry, then refine with this DFT pipeline when accuracy matters.

## Setup (Google Colab)

```python
!pip install -q pyscf geometric py3Dmol
```

Run this once per Colab session before running the pipeline.

## Usage

Paste the contents of `pipeline.py` into a Colab cell and run it. It will:
- Prompt you to upload a `.gjf` file
- Run the optimization (can take minutes to longer, depending on molecule size — DFT is significantly slower than semi-empirical methods like xtb)
- Display the optimized 3D structure inline
- Auto-download a ZIP with all results

Or run locally (with `pyscf`, `geometric`, and `py3Dmol` installed, and without the Colab upload/download calls):

```bash
python pipeline.py
```

## Notes

- Multiplicity is converted to PySCF's `spin` parameter automatically (`spin = multiplicity - 1`).
- Open-shell systems (e.g. high-spin Fe complexes) are handled by switching to unrestricted Kohn-Sham (UKS) when multiplicity ≠ 1.
- Full SCF/optimization output is saved to `optimization.log` for convergence checking and troubleshooting.
- DFT optimizations run on CPU by default in this script; performance will vary significantly with molecule size and Colab's allocated resources.

## License

MIT 
