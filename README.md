<div align="center">

# ⚛️ GJF → DFT Optimization Pipeline

### B3LYP / LANL2DZ geometry optimization, automated end-to-end in Google Colab

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![PySCF](https://img.shields.io/badge/PySCF-DFT-green)
![geomeTRIC](https://img.shields.io/badge/geomeTRIC-optimizer-orange)
![Colab](https://img.shields.io/badge/Google%20Colab-ready-yellow?logo=googlecolab)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

<img src="assets/pipeline_banner.png" alt="GJF to DFT optimization pipeline overview" width="700">

</div>

---

## Overview

A Google Colab pipeline that takes a Gaussian input file (`.gjf`) and runs a full DFT geometry optimization using [PySCF](https://pyscf.org/) and [geomeTRIC](https://github.com/leeping/geomeTRIC) — including support for transition metals like **Fe** and **Zn** via LANL2DZ effective core potentials (ECPs).

No local installs, no manual setup — upload a `.gjf`, get back an optimized structure.

---

## ✨ What it does

| Step | Description |
|------|-------------|
| 1️⃣ Parse | Reads atoms, charge, and multiplicity directly from the `.gjf` file |
| 2️⃣ Build | Constructs a PySCF molecule with per-element LANL2DZ basis + ECP assignment |
| 3️⃣ Optimize | Runs B3LYP geometry optimization via geomeTRIC — auto-switches to unrestricted DFT (UKS) for open-shell/high-spin systems |
| 4️⃣ Visualize | Renders the optimized structure in interactive 3D, inline in the notebook (via py3Dmol) |
| 5️⃣ Export | Packages everything into a downloadable ZIP |

**ZIP contents:**
- `optimized.xyz`
- `optimized.gjf`
- `optimization.log`

---

## 🧪 Why B3LYP/LANL2DZ

B3LYP is a well-validated, general-purpose hybrid functional. LANL2DZ pairs a double-zeta basis for light elements with effective core potentials for heavier atoms, making it a practical default when your molecule includes metals like Fe or Zn — without needing a much heavier all-electron basis set.

> 💡 **Tip:** Use a fast GFN2-xTB pre-optimization first to get a clean starting geometry in seconds, then refine it here with DFT when accuracy actually matters.

---

## 🚀 Setup (Google Colab)

```python
!pip install -q pyscf geometric py3Dmol
```

Run this once per Colab session before running the pipeline.

---

## ▶️ Usage

Paste the contents of `pipeline.py` into a Colab cell and run it. It will:

1. Prompt you to upload a `.gjf` file
2. Run the optimization (minutes to longer, depending on molecule size — DFT is significantly slower than semi-empirical methods like xtb)
3. Display the optimized 3D structure inline
4. Auto-download a ZIP with all results

Or run locally (with `pyscf`, `geometric`, and `py3Dmol` installed, and without the Colab upload/download calls):

```bash
python pipeline.py
```

---

## 📝 Notes

- Multiplicity is converted to PySCF's `spin` parameter automatically (`spin = multiplicity - 1`).
- Open-shell systems (e.g. high-spin Fe complexes) are handled by switching to unrestricted Kohn-Sham (UKS) when multiplicity ≠ 1.
- Full SCF/optimization output is saved to `optimization.log` for convergence checking and troubleshooting.
- DFT optimizations run on CPU by default; performance will vary significantly with molecule size and Colab's allocated resources.

---

## 📄 License

MIT
