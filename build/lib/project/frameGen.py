import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import networkx as nx
from rdkit.Chem import Draw
import pandas as pd
import patchworklib as pw

frame_str = "{a1} {a2} {a3} BO ! atom indices involved in frame {frame} ! {frameid}"
charge_str = "{nchg} 0      ! no. chgs and polarizabilities for atom {atomNumber} ({idx})"
blank_lines = "     0.000000        0.000000        0.000000       1.00000"

from paths import pdb_path


def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   atomic_num=atom.GetAtomicNum(),
                   is_aromatic=atom.GetIsAromatic(),
                   atom_symbol=atom.GetSymbol())

    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())

    return G


def draw_mol_nx(Gmol, labels=None, ax=None):
    Gmol_atom = nx.get_node_attributes(Gmol, 'atom_symbol')
    Gmol_atom = {k: f"{k}.{v}" for k, v in list(Gmol_atom.items())}
    color_map = {'C': 'orange',
                 'O': 'red',
                 'N': 'blue',
                 'S': 'yellow',
                 'H': "grey"}

    _colors = []
    for idx in Gmol.nodes():
        if (Gmol.nodes[idx]['atom_symbol'] in color_map):
            _colors.append(color_map[Gmol.nodes[idx]['atom_symbol']])

    return nx.draw(Gmol,
                   pos=nx.kamada_kawai_layout(Gmol),
                   labels=Gmol_atom if labels is None else labels,
                   with_labels=True,
                   node_color=_colors,
                   node_size=800,
                   ax=ax)


def map_fp_bits_to_atoms(mol, fp):
    bit_info = {}
    AllChem.GetMorganFingerprint(mol, radius=1, bitInfo=bit_info)
    return {bit: list(atoms) for bit, atoms in bit_info.items()}


def fingerprints(molecule):
    AllChem.Compute2DCoords(molecule)
    Chem.SanitizeMol(molecule)
    # molecule = Chem.RemoveHs(molecule)
    fp = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=1)
    atom_maps = map_fp_bits_to_atoms(molecule, fp)
    bi = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(molecule, radius=0, bitInfo=bi)
    bits = list(bi.keys())
    pngs = []
    for bit in bits:
        hits = bi[bit]
        hit_atoms = set()
        hit_bonds = set()
        for atom_idx, radius in hits:
            if radius == 0:
                hit_atoms.add(atom_idx)
            elif radius == 1:
                bond = molecule.GetBondBetweenAtoms(atom_idx, hits[0][0])
                if bond:
                    hit_bonds.add(bond.GetIdx())
        # Draw the molecule with highlighted atoms and bonds
        d = Draw.MolDraw2DCairo(300, 300)  # Replace with appropriate drawer for your environment
        d.drawOptions().prepareMolsBeforeDrawing = False
        Draw.PrepareAndDrawMolecule(d, molecule, highlightAtoms=hit_atoms, highlightBonds=hit_bonds)
        d.FinishDrawing()
        from IPython.display import Image
        png = d.GetDrawingText()
        pngs.append(png)

    return pngs, bits, bi


def make_bit_dict(bi):
    bi_dict = {}
    for k, b in bi.items():
        for t in b:
            bi_dict[t[0]] = k
    return bi_dict


# Function to find all paths of length 3 (three nodes)
def find_paths_of_length_three(G):
    paths = []
    atomnames = nx.get_node_attributes(G, 'atom_symbol')
    for source in G.nodes():
        for target in G.nodes():
            if source != target:
                if atomnames[target] != "H" and atomnames[source] != "H":
                    for path in nx.all_simple_paths(G, source=source, target=target, cutoff=2):
                        if len(path) == 3:
                            paths.append(path)
    return paths


def make_frames(graphMol, triplets, frametypes):
    frames = []
    frame_types = []
    atomnames = nx.get_node_attributes(graphMol, 'atom_symbol')
    heavy_nodes = [_ for _ in range(len(atomnames)) if atomnames[_] != 'H']
    hydrogens = [_ for _ in range(len(atomnames)) if atomnames[_] == 'H']
    while len(heavy_nodes) != 0:
        #  add frames by frame priority
        for pf in frame_priority:
            for i, atypframe in enumerate(frametypes):
                if atypframe == pf:
                    frames.append(triplets[i])
                    frame_types.append(pf)
                    # remove them from the nodes list
                    for a in triplets[i]:
                        if a in heavy_nodes:
                            heavy_nodes.remove(a)
    return frames, frame_types, hydrogens


def make_frame_file(resName, frames, frame_types, hydrogens):
    print("1 0          ! no. residue types defined here")
    print("")
    print(f"{resName} !residue name")
    print(f"{len(frames)} ! no. axis system frames")
    count_charges = 0
    atoms_added = []
    for i, (frame, frametype) in enumerate(zip(frames, frame_types)):
        if not np.all([a in atoms_added for a in frame]):
            print(frame_str.format(a1=frame[0] + 1, a2=frame[1] + 1, a3=frame[2] + 1, frame=i + 1, frameid=frametype))
            for j, a in enumerate(frame):
                if a not in atoms_added:
                    print(charge_str.format(nchg=2, atomNumber=a + 1, idx=j))
                    print(blank_lines)
                    print(blank_lines)
                    atoms_added.append(a)
                    count_charges += 2
                else:
                    print(charge_str.format(nchg=0, atomNumber=a + 1, idx=j))
    frameCount = i
    for h in hydrogens:
        print(frame_str.format(a1=h + 1, a2=0, a3=0, frame=frameCount + 1, frameid="(hydrogen)"))
        print(charge_str.format(nchg=1, atomNumber=h + 1, idx=0))
        print(blank_lines)
        print("0 0      ! no. chgs and polarizabilities for atom NULL (0)")
        print("0 0      ! no. chgs and polarizabilities for atom NULL (0)")
        atoms_added.append(h)
        frameCount += i
        count_charges += 1

    print("!")
    print("!ncghs", count_charges)
    atoms_added.sort()
    print("!", atoms_added)
    print("!natoms", len(atoms_added))

sdf_files = list(pdb_path.glob("*-min.pdb.sdf"))
mols = []
names = []
for sdf_file in sdf_files:
    m = next(Chem.SDMolSupplier(str(sdf_file),  sanitize=True, removeHs=False))
    print(sdf_file, m)
    if m is not None:
        mols.append(m)
        names.append(str(sdf_file.stem).split("-")[0].upper())
    else:
        sdf_files.remove(sdf_file)

formal_charges = [rdkit.Chem.rdmolops.GetFormalCharge(m) for m in mols]
legends = [f"{n} ({fc})" for fc, n in zip(formal_charges, names)]
Gmols = [mol_to_nx(mol) for mol in mols]

pictures = []
all_bits = []
bi_dicts = []
for mol in mols:
    pngs, bits, bi = fingerprints(mol)
    pictures.append(pngs)
    all_bits.append(bits)
    bi_dicts.append(make_bit_dict(bi))

axes = []
for i in range(len(Gmols)):
    ax = pw.Brick(figsize=(5, 5))
    draw_mol_nx(Gmols[i], labels=bi_dicts[i], ax=ax)
    axes.append(ax)

paths3 = []
results = []

for i, G in enumerate(Gmols):
    # Find paths of length 3 in our sample graph
    paths_length_three = find_paths_of_length_three(Gmols[i])
    res = [tuple([bi_dicts[i][x] for x in _]) for _ in paths_length_three]
    paths3.append(paths_length_three)
    results.append(res)

#  find all possible frame combinations and sort by frequency
all_frame_types = pd.DataFrame(results).to_numpy().flatten()
unique_frames = pd.DataFrame(all_frame_types).value_counts()
unique_frames = list(unique_frames.index)
frame_priority = [unique_frames[0][0]]
for frame in unique_frames[1:]:
    #  check if the palindrome is not there
    if frame[0][-1::-1] not in frame_priority:
        frame_priority.append(frame[0])

