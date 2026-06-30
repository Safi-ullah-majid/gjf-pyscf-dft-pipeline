# =====================================================================
# GJF -> DFT Geometry Optimization Pipeline (PySCF + geomeTRIC, Colab)
# Functional: B3LYP   |   Basis/ECP: LANL2DZ (supports Fe, Zn, and most
# transition metals via effective core potentials)
# =====================================================================
# Usage in Colab:
#   1. Run CELL 1 (install) once.
#   2. Run CELL 2 — upload your .gjf, it parses charge/multiplicity/
#      coordinates automatically, runs the DFT optimization, shows a
#      3D view of the optimized structure, and downloads a ZIP with:
#        - optimized.xyz
#        - optimized.gjf
#        - optimization.log
# =====================================================================

# --------------------- CELL 1: SETUP (run once) ---------------------
# !pip install -q pyscf geometric py3Dmol
# ----------------------------------------------------------------------

import os
import re
import shutil
import zipfile
from datetime import datetime

PERIODIC_ELEMENTS = {
    "h","he","li","be","b","c","n","o","f","ne","na","mg","al","si","p","s",
    "cl","ar","k","ca","sc","ti","v","cr","mn","fe","co","ni","cu","zn","ga",
    "ge","as","se","br","kr","rb","sr","y","zr","nb","mo","tc","ru","rh","pd",
    "ag","cd","in","sn","sb","te","i","xe","cs","ba","la","ce","pr","nd","pm",
    "sm","eu","gd","tb","dy","ho","er","tm","yb","lu","hf","ta","w","re","os",
    "ir","pt","au","hg","tl","pb","bi","po","at","rn"
}

# Elements that need an effective core potential (ECP) under LANL2DZ.
# PySCF's 'lanl2dz' basis already implies the matching ECP for these.
ECP_ELEMENTS = {
    "k","ca","sc","ti","v","cr","mn","fe","co","ni","cu","zn","ga","ge","as",
    "se","br","kr","rb","sr","y","zr","nb","mo","tc","ru","rh","pd","ag","cd",
    "in","sn","sb","te","i","xe","cs","ba","la","ce","pr","nd","pm","sm","eu",
    "gd","tb","dy","ho","er","tm","yb","lu","hf","ta","w","re","os","ir","pt",
    "au","hg","tl","pb","bi","po","at","rn"
}


def parse_gjf(path):
    with open(path, "r") as f:
        lines = f.readlines()

    blocks, current = [], []
    for line in lines:
        if line.strip() == "":
            blocks.append(current)
            current = []
        else:
            current.append(line.rstrip("\n"))
    if current:
        blocks.append(current)
    blocks = [b for b in blocks if b]

    coord_block = None
    for b in blocks:
        if re.match(r"^\s*-?\d+\s+\d+\s*$", b[0]):
            coord_block = b
            break
    if coord_block is None:
        raise ValueError("Could not find charge/multiplicity + coordinate block in .gjf")

    charge, mult = coord_block[0].split()
    charge, mult = int(charge), int(mult)

    atoms = []
    for line in coord_block[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        elem = re.sub(r"[^A-Za-z]", "", parts[0])
        if elem.lower() not in PERIODIC_ELEMENTS:
            continue
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue
        atoms.append((elem, x, y, z))

    if not atoms:
        raise ValueError("No atoms parsed from .gjf — check formatting.")

    return charge, mult, atoms


def write_xyz(atoms, path, comment="Generated"):
    with open(path, "w") as f:
        f.write(f"{len(atoms)}\n{comment}\n")
        for elem, x, y, z in atoms:
            f.write(f"{elem:<3} {x:>14.8f} {y:>14.8f} {z:>14.8f}\n")


def write_gjf(atoms, charge, mult, path, route="#p b3lyp/lanl2dz opt", title="DFT optimized structure"):
    with open(path, "w") as f:
        f.write(f"%chk={os.path.splitext(os.path.basename(path))[0]}.chk\n")
        f.write(f"{route}\n\n{title}\n\n{charge} {mult}\n")
        for elem, x, y, z in atoms:
            f.write(f"{elem:<3} {x:>14.8f} {y:>14.8f} {z:>14.8f}\n")
        f.write("\n")


def build_pyscf_mol(atoms, charge, mult):
    from pyscf import gto

    atom_str = "\n".join(f"{el} {x:.8f} {y:.8f} {z:.8f}" for el, x, y, z in atoms)
    spin = mult - 1  # PySCF spin = n_alpha - n_beta = multiplicity - 1

    # Per-element basis/ECP assignment: LANL2DZ everywhere, with ECP
    # automatically pulled in by PySCF for transition metals / heavy atoms.
    basis = {}
    ecp = {}
    for el, *_ in atoms:
        key = el if el[0].isupper() else el.capitalize()
        basis[key] = "lanl2dz"
        if el.lower() in ECP_ELEMENTS:
            ecp[key] = "lanl2dz"

    mol = gto.M(
        atom=atom_str,
        basis=basis,
        ecp=ecp if ecp else None,
        charge=charge,
        spin=spin,
        verbose=4,
    )
    return mol


def run_dft_optimization(atoms, charge, mult, workdir, log_path):
    from pyscf import dft
    from pyscf.geomopt.geometric_solver import optimize

    mol = build_pyscf_mol(atoms, charge, mult)

    mf = dft.KS(mol)
    mf.xc = "b3lyp"
    if mult != 1:
        mf = mf.to_uks()  # unrestricted for open-shell systems (e.g. high-spin Fe)

    with open(log_path, "w") as logf:
        mol.stdout = logf
        mol.verbose = 4

        mol_eq = optimize(mf, maxsteps=100)

    opt_atoms = []
    for (sym, _), coord in zip(mol_eq._atom, mol_eq.atom_coords(unit="Angstrom")):
        opt_atoms.append((sym, float(coord[0]), float(coord[1]), float(coord[2])))

    return opt_atoms


def show_3d(xyz_path):
    import py3Dmol
    with open(xyz_path) as f:
        xyz_data = f.read()
    view = py3Dmol.view(width=500, height=400)
    view.addModel(xyz_data, "xyz")
    view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
    view.zoomTo()
    view.show()


def pipeline(gjf_path, workdir="dft_workdir"):
    os.makedirs(workdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(gjf_path))[0]

    print(f"[1/5] Parsing {gjf_path} ...")
    charge, mult, atoms = parse_gjf(gjf_path)
    print(f"   Charge={charge}, Multiplicity={mult}, Atoms={len(atoms)}")

    print("[2/5] Running B3LYP/LANL2DZ geometry optimization (this can take a few minutes)...")
    log_path = os.path.join(workdir, "optimization.log")
    opt_atoms = run_dft_optimization(atoms, charge, mult, workdir, log_path)
    print(f"   Done. Log: {log_path}")

    print("[3/5] Writing optimized outputs ...")
    opt_xyz = os.path.join(workdir, "optimized.xyz")
    opt_gjf = os.path.join(workdir, "optimized.gjf")
    write_xyz(opt_atoms, opt_xyz, comment=f"B3LYP/LANL2DZ optimized geometry of {base}")
    write_gjf(opt_atoms, charge, mult, opt_gjf,
              route="#p b3lyp/lanl2dz opt", title=f"B3LYP/LANL2DZ optimized: {base}")

    print("[4/5] Rendering 3D structure ...")
    show_3d(opt_xyz)

    print("[5/5] Packaging ZIP ...")
    zip_path = f"{base}_dft_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(opt_xyz, arcname="optimized.xyz")
        zf.write(opt_gjf, arcname="optimized.gjf")
        zf.write(log_path, arcname="optimization.log")

    print(f"\nFinished. ZIP ready: {zip_path}")
    return zip_path


# --------------------- CELL 2: RUN (upload + execute) ---------------------
if __name__ == "__main__":
    from google.colab import files

    print("Upload your .gjf file:")
    uploaded = files.upload()
    gjf_file = list(uploaded.keys())[0]

    zip_path = pipeline(gjf_file)
    files.download(zip_path)
