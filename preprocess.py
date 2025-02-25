import os
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from rdkit import Chem
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
import nglview as nv
from rdkit.Chem.rdchem import ChiralType, HybridizationType

###############################
# Node Feature Extraction
###############################

def enrich_node_features(ligand_mol, elements):
    """
    Node features for ligand molecule.
      - One-hot encoding of the atom type.
      - Chirality (2-dim one-hot; [1,0] for CW, [0,1] for CCW, [0,0] if not chiral).
      - Hybridization (one-hot: SP, SP2, SP3, Other).
      - Aromaticity (binary).
      - Ring membership (binary).
    """
    def one_hot_encode(value, allowed_values):
        vec = np.zeros(len(allowed_values))
        try:
            idx = allowed_values.index(value)
            vec[idx] = 1
        except ValueError as e:
            raise ValueError(f"Invalid atom type: {value}")
        return vec

    def get_chirality(atom):
        tag = atom.GetChiralTag()
        if tag == ChiralType.CHI_TETRAHEDRAL_CW:
            return np.array([1, 0])
        elif tag == ChiralType.CHI_TETRAHEDRAL_CCW:
            return np.array([0, 1])
        else:
            return np.array([0, 0])
    
    def get_hybridization(atom):
        hyb = atom.GetHybridization()
        if hyb == HybridizationType.SP:
            return np.array([1, 0, 0, 0])
        elif hyb == HybridizationType.SP2:
            return np.array([0, 1, 0, 0])
        elif hyb == HybridizationType.SP3:
            return np.array([0, 0, 1, 0])
        else:
            return np.array([0, 0, 0, 1])
    
    enriched_features = []
    for atom in ligand_mol.GetAtoms():
        atom_type_feat = one_hot_encode(atom.GetSymbol(), elements)
        chirality_feat = get_chirality(atom)
        hybridization_feat = get_hybridization(atom)
        aromatic_feat = np.array([1]) if atom.GetIsAromatic() else np.array([0])
        ring_feat = np.array([1]) if atom.IsInRing() else np.array([0])
        node_feature = np.concatenate([
            atom_type_feat,
            chirality_feat,
            hybridization_feat,
            aromatic_feat,
            ring_feat
        ])
        enriched_features.append(node_feature)
    return np.array(enriched_features)

###############################
# Edge Feature Extraction
###############################

def get_covalent_edge_features(bond):
    """
    Edge feature vector.
    
    Covalent part (first 8 dims):
      - Bond type as one-hot (order: single, double, triple, aromatic, misc) -> 5 dims.
      - Stereochemistry (obtained from get_bond_stereo_feature) -> 2 dims.
      - Conjugation (binary) -> 1 dim.
      
    Non-covalent part (last 4 dims):
      - Set to zeros.
    """
    # Bond type one-hot encoding
    bt = bond.GetBondType()
    if bt == Chem.rdchem.BondType.SINGLE:
        bond_type = [1, 0, 0, 0, 0]
    elif bt == Chem.rdchem.BondType.DOUBLE:
        bond_type = [0, 1, 0, 0, 0]
    elif bt == Chem.rdchem.BondType.TRIPLE:
        bond_type = [0, 0, 1, 0, 0]
    elif bt == Chem.rdchem.BondType.AROMATIC:
        bond_type = [0, 0, 0, 1, 0]
    else:
        bond_type = [0, 0, 0, 0, 1]
    
    # Stereochemistry using RDKit
    def get_bond_stereo_feature(bond):
        stereo_enum = bond.GetStereo()
        if stereo_enum == Chem.rdchem.BondStereo.STEREOZ:
            return np.array([1, 0])
        elif stereo_enum == Chem.rdchem.BondStereo.STEREOE:
            return np.array([0, 1])
        else:
            return np.array([0, 0])
    
    stereo = get_bond_stereo_feature(bond)
    conjugation = np.array([1]) if bond.GetIsConjugated() else np.array([0])
    covalent_feats = np.concatenate([bond_type, stereo, conjugation])
    noncovalent_feats = np.zeros(4)
    return np.concatenate([covalent_feats, noncovalent_feats])

def is_hydrogen_bond(atom1_sym, atom2_sym, pos1, pos2):
    """
    Hydrogen bond:
      - Distance < 3.5 Å.
      - Both atoms are electronegative (e.g., N, O, F).
    """
    electronegative = {'N', 'O', 'F'}
    distance = np.linalg.norm(pos1 - pos2)
    return distance < 3.5 and (atom1_sym in electronegative and atom2_sym in electronegative)

def is_hydrophobic_interaction(atom1_sym, atom2_sym, pos1, pos2):
    """
    Hydrophobic interaction:
      - Both atoms are carbon.
      - Distance < 5.0 Å.
    """
    distance = np.linalg.norm(pos1 - pos2)
    return atom1_sym == 'C' and atom2_sym == 'C' and distance < 5.0

def is_ionic_interaction(atom1_sym, atom2_sym, pos1, pos2, charge1, charge2):
    """
    Ionic interaction:
      - Distance < 4.0 Å.
      - One atom is positive and the other negative.
    """
    distance = np.linalg.norm(pos1 - pos2)
    return distance < 4.0 and (charge1 * charge2 < 0)

def is_aromatic_interaction(atom1_sym, atom2_sym, pos1, pos2, arom1, arom2):
    """
    Aromatic interaction:
      - Both atoms are aromatic.
      - Distance < 5.0 Å.
    """
    distance = np.linalg.norm(pos1 - pos2)
    return arom1 and arom2 and distance < 5.0

def get_noncovalent_edge_features_custom(atom1_sym, atom2_sym, pos1, pos2, charge1, charge2, arom1, arom2):
    """ 
    The first 8 dims (for covalent features) are zeros.
    The last 4 dims encode flags for:
      - Hydrogen bond
      - Hydrophobic interaction
      - Ionic interaction
      - Aromatic interaction
    """
    hb = 1 if is_hydrogen_bond(atom1_sym, atom2_sym, pos1, pos2) else 0
    hydrophob = 1 if is_hydrophobic_interaction(atom1_sym, atom2_sym, pos1, pos2) else 0
    ionic = 1 if is_ionic_interaction(atom1_sym, atom2_sym, pos1, pos2, charge1, charge2) else 0
    aromatic = 1 if is_aromatic_interaction(atom1_sym, atom2_sym, pos1, pos2, arom1, arom2) else 0
    return np.concatenate([np.zeros(8), np.array([hb, hydrophob, ionic, aromatic])])

def get_noncovalent_edge_features():
    """
    Fallback dummy 12-dim feature vector for non-covalent interactions.
    """
    return np.zeros(12)

def enrich_edge_features(ligand_mol, edge_index_list, n_lig, ligand_coords, protein_coords, protein_atoms):
    """
    Compute edge features for each edge in edge_index_list.
    
    Cases:
      1. Both nodes are ligand atoms and a covalent bond exists: extract covalent features.
      2. Both nodes are ligand atoms but no covalent bond exists: extract non-covalent features using ligand properties.
      3. One node is ligand and the other is protein: use ligand properties from RDKit and protein info from Prot.
      4. Both nodes are protein atoms: return a dummy non-covalent 12-dim vector.
    """
    edge_features = []
    for edge in edge_index_list:
        i, j = edge
        # Case 1 and 2: both nodes are ligand atoms
        if i < n_lig and j < n_lig:
            bond = ligand_mol.GetBondBetweenAtoms(i, j)
            if bond is not None:
                feat = get_covalent_edge_features(bond)
            else:
                atom1 = ligand_mol.GetAtomWithIdx(i)
                atom2 = ligand_mol.GetAtomWithIdx(j)
                atom1_sym = atom1.GetSymbol()
                atom2_sym = atom2.GetSymbol()
                pos1 = ligand_coords[i]
                pos2 = ligand_coords[j]
                charge1 = atom1.GetFormalCharge()
                charge2 = atom2.GetFormalCharge()
                arom1 = atom1.GetIsAromatic()
                arom2 = atom2.GetIsAromatic()
                feat = get_noncovalent_edge_features_custom(atom1_sym, atom2_sym, pos1, pos2, charge1, charge2, arom1, arom2)
        # Case 3: one ligand and one protein node
        elif (i < n_lig and j >= n_lig) or (i >= n_lig and j < n_lig):
            if i < n_lig:
                # i: ligand, j: protein
                ligand_atom = ligand_mol.GetAtomWithIdx(i)
                ligand_sym = ligand_atom.GetSymbol()
                pos1 = ligand_coords[i]
                charge1 = ligand_atom.GetFormalCharge()
                arom1 = ligand_atom.GetIsAromatic()
                prot_idx = j - n_lig
                protein_sym = protein_atoms[prot_idx]
                pos2 = protein_coords[prot_idx]
                # Assume default protein charge and aromatic flag (or set them to 0)
                charge2 = 0
                arom2 = 0
                feat = get_noncovalent_edge_features_custom(ligand_sym, protein_sym, pos1, pos2, charge1, charge2, arom1, arom2)
            else:
                # i: protein, j: ligand
                prot_idx = i - n_lig
                protein_sym = protein_atoms[prot_idx]
                pos1 = protein_coords[prot_idx]
                charge1 = 0
                arom1 = 0
                ligand_atom = ligand_mol.GetAtomWithIdx(j)
                ligand_sym = ligand_atom.GetSymbol()
                pos2 = ligand_coords[j]
                charge2 = ligand_atom.GetFormalCharge()
                arom2 = ligand_atom.GetIsAromatic()
                feat = get_noncovalent_edge_features_custom(ligand_sym, protein_sym, pos2, pos1, charge2, charge1, arom2, arom1)
        # Case 4: both nodes are protein atoms
        else:
            feat = get_noncovalent_edge_features()
        edge_features.append(feat)
    return np.array(edge_features)

###############################
# Complex Processing
###############################

def process_complex(ligand_path, protein_path, elements, cutoff=5.0):
    """
    Process a ligand-protein complex.
    
    Loads the ligand (SDF) and protein (PDB), computes ligand coordinates,
    enriched node features, and identifies pocket residues (within cutoff).
    """
    ligand = Chem.SDMolSupplier(ligand_path)[0]
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", protein_path)

    # Ligand processing
    ligand_atoms = [atom.GetSymbol() for atom in ligand.GetAtoms()]
    ligand_coords = np.array([list(ligand.GetConformer(0).GetAtomPosition(idx)) 
                              for idx in range(ligand.GetNumAtoms())])
    ligand_one_hot = enrich_node_features(ligand, elements)

    # Identify interacting pocket residues
    pocket_residues = []
    for residue in structure[0].get_residues():
        protein_coords = np.array([atom.get_coord() for atom in residue.get_atoms()])
        if is_aa(residue.get_resname(), standard=True):
            if (((protein_coords[:, None, :] - ligand_coords[None, :, :]) ** 2).sum(-1) ** 0.5).min() < cutoff:
                pocket_residues.append(residue)
    pocket_ids = [f'{res.parent.id}:{res.id[1]}' for res in pocket_residues]

    # Pocket processing: get all atoms from selected residues
    pocket_coords = np.concatenate(
        [np.array([atom.get_coord() for atom in residue.get_atoms()]) 
         for residue in pocket_residues], axis=0)
    pocket_atom = np.concatenate(
        [np.array([atom.element for atom in residue.get_atoms()]) 
         for residue in pocket_residues], axis=0)

    # Filter pocket atoms by allowed elements
    approved_mask = np.isin(pocket_atom, elements)
    pocket_atom = pocket_atom[approved_mask]
    pocket_coords = pocket_coords[approved_mask]

    # Use a simple one-hot encoding for pocket atoms; pad with zeros for extra dims.
    pocket_one_hot = np.zeros((len(pocket_atom), len(elements) + 8))
    for idx, atom in enumerate(pocket_atom):
        pocket_one_hot[idx, elements.index(atom)] = 1

    # Center coordinates on the pocket's center of mass
    pocket_center = pocket_coords.mean(axis=0)
    pocket_coords -= pocket_center
    ligand_coords = ligand_coords - pocket_center

    ligand_data = {
        'lig_coords': ligand_coords,
        'lig_one_hot': ligand_one_hot,
    }
    # Include pocket_atom in pocket_data so we have protein element info.
    pocket_data = {
        'pocket_coords': pocket_coords,
        'pocket_one_hot': pocket_one_hot,
        'pocket_ids': pocket_ids,
        'pocket_atom': pocket_atom
    }
    return ligand_data, pocket_data, ligand

###############################
# Graph Creation with Enriched Edge Features
###############################

def create_enriched_graph(ligand_data, protein_data, ligand_mol, cutoff=3.0):
    """
    Build graph with enriched node and edge features.
      
    Returns:
      data: a torch_geometric.data.Data object with:
            - x: node features
            - pos: node positions
            - edge_index: connectivity
            - edge_attr: enriched edge features
    """
    ligand_coords = ligand_data['lig_coords']
    ligand_features = ligand_data['lig_one_hot']
    pocket_coords = protein_data['pocket_coords']
    pocket_features = protein_data['pocket_one_hot']
    protein_atoms = protein_data['pocket_atom']

    n_lig = ligand_coords.shape[0]
    n_pocket = pocket_coords.shape[0]

    edge_index_list = []
    
    # Ligand-ligand edges
    for i in range(n_lig):
        for j in range(i+1, n_lig):
            d = np.linalg.norm(ligand_coords[i] - ligand_coords[j])
            if d < cutoff:
                edge_index_list.append([i, j])
                edge_index_list.append([j, i])
    # Protein-protein edges
    for i in range(n_pocket):
        for j in range(i+1, n_pocket):
            d = np.linalg.norm(pocket_coords[i] - pocket_coords[j])
            if d < cutoff:
                edge_index_list.append([n_lig + i, n_lig + j])
                edge_index_list.append([n_lig + j, n_lig + i])
    # Cross edges (ligand-protein)
    for i in range(n_lig):
        for j in range(n_pocket):
            d = np.linalg.norm(ligand_coords[i] - pocket_coords[j])
            if d < cutoff:
                edge_index_list.append([i, n_lig + j])
                edge_index_list.append([n_lig + j, i])
    
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    
    # Enriched edge features: pass both ligand and protein information
    edge_attr_np = enrich_edge_features(ligand_mol, edge_index_list, n_lig, ligand_coords, pocket_coords, protein_atoms)
    edge_attr = torch.tensor(edge_attr_np, dtype=torch.float)
    
    combined_features = np.concatenate([ligand_features, pocket_features], axis=0)
    combined_coords = np.concatenate([ligand_coords, pocket_coords], axis=0)
    
    data = Data(
        x=torch.tensor(combined_features, dtype=torch.float),
        pos=torch.tensor(combined_coords, dtype=torch.float),
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    return data

###############################
# Main Script
###############################

ELEMENTS = ["C", "N", "O", "P", "S", "F", "Cl", "Br", "I"]

base_dir = "output"
protein_paths = []
ligand_paths = []
complex_paths = []

for entry in os.scandir(base_dir):
    if entry.is_dir():
        dirname = entry.name
        dirpath = entry.path
        receptor_filename = f"{dirname}_receptor.pdb"
        ligand_filename = f"{dirname}_ligand.sdf"
        complex_filename = f"{dirname}.pdb"
        receptor_path = os.path.join(dirpath, receptor_filename)
        ligand_path = os.path.join(dirpath, ligand_filename)
        complex_path = os.path.join(dirpath, complex_filename)
        if os.path.exists(receptor_path) and os.path.exists(ligand_path) and os.path.exists(complex_path):
            protein_paths.append(receptor_path)
            ligand_paths.append(ligand_path)
            complex_paths.append(complex_path)
        else:
            print(f"Missing receptor or ligand file in {dirpath}")

print("Protein Paths:")
for p in protein_paths:
    print(p)
print("\nLigand Paths:")
for l in ligand_paths:
    print(l)

graphs = []
for prot, lig in zip(protein_paths, ligand_paths):
    # Process complex and retrieve ligand_data, pocket_data, and ligand_mol
    Lig, Prot, ligand_mol = process_complex(lig, prot, ELEMENTS, cutoff=5.0)
    print(f"Processed {prot} and {lig}")
    graph = create_enriched_graph(Lig, Prot, ligand_mol, cutoff=3.0)
    if graph is not None:
        graphs.append(graph)
print(f"Generated {len(graphs)} valid graphs out of {len(protein_paths)} protein files.")

with open("graphs.pkl", "wb") as f:
    pickle.dump(graphs, f)
print("All graphs saved to graphs.pkl")
