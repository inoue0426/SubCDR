import numpy as np
import pandas as pd
import torch

from molFrags import *


def data_process():
    # --------data_load
    PATH = "./nci_data"
    Drug_file = "%s/drug_smiles.csv" % PATH
    Cell_line_file = "%s/cell line_GEP.csv" % PATH
    Gene_role_file = "%s/gene_role.csv" % PATH
    Mask_file = "%s/masked.csv" % PATH
    # --------data_preprocessing
    # ---drugs preprocessing
    drug = pd.read_csv(Drug_file, sep=",", header=0, index_col=[0])
    # ---get fragment features for all drug smiles
    drug_subfeat = {}
    drug_fragments = {}
    SMARTS = []
    max_len = 0
    for tup in zip(drug["NSC"], drug["SMILES"]):
        # ---smiles to frags
        sub_smi, sm = BRICS_GetMolFrags(tup[1])
        max_len = len(sub_smi) if len(sub_smi) > max_len else max_len
        # ---mols to fingerprints
        sub_features = [np.array(get_Morgan(item)) for item in sub_smi]
        drug_subfeat[str(tup[0])] = np.array(sub_features)
        SMARTS.append(sm)
        drug_fragments[str(tup[0])] = sub_smi
    drug_dim = drug_subfeat["740"].shape[1]

    # ---cell lines preprocessing
    gexpr_data = pd.read_csv(Cell_line_file, sep=",", header=0, index_col=[0])
    Mask = pd.read_csv(Mask_file, sep=",", header=0, index_col=[0])
    gexpr_data = gexpr_data * Mask
    gene_annotation = pd.read_csv(Gene_role_file, sep=",", header=0, index_col=[0])
    gene_types = list(set(gene_annotation["Role in Cancer"]))
    cline_subfeat = {}
    type_count = gene_annotation["Role in Cancer"].value_counts()
    cline_dim = max(type_count)
    # ---get fragments for all cell line expressions
    for index, row in gexpr_data.iterrows():
        sub_gexpr = []
        for gt in gene_types:
            gt_gexpr = row[gene_annotation["Role in Cancer"] == gt]
            # ---padding
            value = gt_gexpr.values
            padding = np.zeros((cline_dim - len(value)))
            sub_gexpr.append(
                list(
                    np.concatenate((value, padding), axis=0)
                    if len(value) < cline_dim
                    else value
                )
            )
        cline_subfeat[str(index)] = np.array(sub_gexpr)

    return drug_subfeat, cline_subfeat, drug_dim, drug_fragments, gene_types
