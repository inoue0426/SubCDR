import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import KFold

from MF import *
from models_classify import *
from nci_data_process import data_process
from torch_dataset import *
from utils import *

parser = argparse.ArgumentParser(
    description="Cancer_Drug_Response_Prediction_Independent"
)
parser.add_argument(
    "--lr",
    dest="lr",
    type=float,
    default=0.0001,
)
parser.add_argument("--batch_sizes", dest="bs", type=int, default=50)
parser.add_argument("--epoch", dest="ep", type=int, default=100)
parser.add_argument(
    "-output_dir", dest="o", default="./output_dir/", help="output directory"
)
args = parser.parse_args()

os.makedirs(args.o, exist_ok=True)
# ---data process
start_time = time.time()
seed = 2022
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
drug_subfeat, cline_subfeat, drug_dim, drug_compo_elem, cline_compos_elem = (
    data_process()
)

# %%---dataset_split and compile
# train_size = 0.9
# CV, Independent = np.split(
#     CDR_pairs.sample(frac=1, random_state=seed), [int(train_size * len(CDR_pairs))]
# )

CV = pd.read_csv("nci_data/nci_train.csv")
Independent = pd.read_csv("nci_data/nci_valid.csv")
test_Independent = pd.read_csv("nci_data/nci_test.csv")

CV.columns = ["Drug", "Cline", "IC50"]
Independent.columns = ["Drug", "Cline", "IC50"]
test_Independent.columns = ["Drug", "Cline", "IC50"]

CV = CV[["Cline", "Drug", "IC50"]]
Independent = Independent[["Cline", "Drug", "IC50"]]
test_Independent = test_Independent[["Cline", "Drug", "IC50"]]


# ---Binarization of the IC50 values with Z-score normalization (threshold = 0)
def getBinary(Tensors, thresh=0):
    ones = torch.ones_like(Tensors)
    zeros = torch.zeros_like(Tensors)
    return torch.where(Tensors < thresh, ones, zeros)


# ---data batchsize
def PairFeatures(
    pairs, drug_subfeat, cline_subfeat, drug_glofeat, cline_glofeat, drug_compo_elem
):
    drug_subs = []
    cline_subs = []
    drug_glos = []
    cline_glos = []
    drug_compos = []
    cline_compos = []
    label = []
    for _, row in pairs.iterrows():
        cline_subs.append(cline_subfeat[str(row[0])])
        drug_subs.append(drug_subfeat[str(row[1])])
        cline_glos.append(np.array(cline_glofeat.loc[row[0]]))
        drug_glos.append(np.array(drug_glofeat.loc[row[1]]))
        drug_compos.append([row[1], drug_compo_elem[str(row[1])]])
        cline_compos.append([row[0], cline_compos_elem])
        label.append(row[2])
    return (
        drug_subs,
        cline_subs,
        drug_glos,
        cline_glos,
        drug_compos,
        cline_compos,
        label,
    )


def BatchGenerate(
    pairs, drug_subfeat, cline_subfeat, drug_glofeat, cline_glofeat, drug_compo_elem, bs
):
    drug_subs, cline_subs, drug_glos, cline_glos, drug_compos, cline_compos, label = (
        PairFeatures(
            pairs,
            drug_subfeat,
            cline_subfeat,
            drug_glofeat,
            cline_glofeat,
            drug_compo_elem,
        )
    )
    ds_loader = Data.DataLoader(
        BatchData(drug_subs), batch_size=bs, shuffle=False, collate_fn=collate_seq
    )
    cs_loader = Data.DataLoader(BatchData(cline_subs), batch_size=bs, shuffle=False)
    glo_loader = Data.DataLoader(
        PairsData(drug_glos, cline_glos), batch_size=bs, shuffle=False
    )
    label = torch.from_numpy(np.array(label, dtype="float32")).to(device)
    label = Data.DataLoader(
        dataset=Data.TensorDataset(label), batch_size=bs, shuffle=False
    )
    return ds_loader, cs_loader, glo_loader, drug_compos, cline_compos, label


def train(drug_loader_train, cline_loader_train, glo_loader_train, label_train):
    loss_train = 0
    Y_true, Y_pred = [], []
    for batch, (drug, cline, glo_feat, label) in enumerate(
        zip(drug_loader_train, cline_loader_train, glo_loader_train, label_train)
    ):
        label = getBinary(label[0])
        pred, _ = model(drug.to(device), cline.to(device), glo_feat.to(device))
        optimizer.zero_grad()
        loss = myloss(pred, label)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        Y_true += label.cpu().detach().numpy().tolist()
        Y_pred += pred.cpu().detach().numpy().tolist()
    auc, aupr = classification_metric(Y_true, Y_pred)
    print("train-loss=", loss_train / len(CV))
    print("train-AUC:" + str(round(auc, 4)) + " train-AUPR:" + str(round(aupr, 4)))


def test(drug_loader_test, cline_loader_test, glo_loader_test, label_test):
    loss_test = 0
    Y_true, Y_pred = [], []
    all_maps = []
    model.eval()
    with torch.no_grad():
        for batch, (drug, cline, glo_feat, label) in enumerate(
            zip(drug_loader_test, cline_loader_test, glo_loader_test, label_test)
        ):
            label = getBinary(label[0])
            pred, maps = model(drug.to(device), cline.to(device), glo_feat.to(device))
            loss = myloss(pred, label)
            loss_test += loss.item()
            Y_true += label.cpu().detach().numpy().tolist()
            Y_pred += pred.cpu().detach().numpy().tolist()
    print("test-loss=", loss.item() / len(Independent))
    auc, aupr = classification_metric(Y_true, Y_pred)
    return auc, aupr, Y_true, Y_pred


# %%---traing and test
# ---Building known matrix
print("Building known matrix")
CDR_known = CV.set_index(["Cline", "Drug"]).unstack("Cline")
CDR_known.columns = CDR_known.columns.droplevel()
# ---MF
print("MF")
CDR_matrix = np.array(CDR_known)
CDR_mask = 1 - np.float32(np.isnan(CDR_matrix))
CDR_matrix[np.isnan(CDR_matrix)] = 0

print("SVD")
drug_glofeat, cline_glofeat = svt_solve(A=CDR_matrix, mask=CDR_mask)
drug_glofeat = pd.DataFrame(drug_glofeat)
cline_glofeat = pd.DataFrame(cline_glofeat)
drug_glofeat.index = list(CDR_known.index)
cline_glofeat.index = list(CDR_known.columns)
glo_dim = 2 * drug_glofeat.shape[1]

Result = []
# Randomly shuffle samples
print("Randomly shuffle samples")
CV = CV.sample(frac=1, random_state=seed)
Independent = Independent.sample(frac=1, random_state=seed)

batch_sizes = args.bs
print("batch_sizes = %d" % batch_sizes)
# ---data batchsize
print("Generating training data batches...")
drug_loader_train, cline_loader_train, glo_loader_train, _, _, label_train = (
    BatchGenerate(
        CV,
        drug_subfeat,
        cline_subfeat,
        drug_glofeat,
        cline_glofeat,
        drug_compo_elem,
        bs=batch_sizes,
    )
)
print("Generating test data batches...")
drug_loader_test, cline_loader_test, glo_loader_test, dc_test, cc_test, label_test = (
    BatchGenerate(
        Independent,
        drug_subfeat,
        cline_subfeat,
        drug_glofeat,
        cline_glofeat,
        drug_compo_elem,
        bs=batch_sizes,
    )
)

# %%
print("Initializing model...")
model = SubCDR(
    SubEncoder(in_drug=drug_dim, in_cline=8, out=82),
    GraphEncoder(in_channels=32, out_channels=16),
    GloEncoder(in_channels=glo_dim, out_channels=128),
    Decoder(in_channels=160),
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
print(f"Learning rate: {args.lr}")
myloss = torch.nn.BCELoss()

# ---main
print("Starting training...")
Result = []
start = time.time()
final_AUC = 0
final_AUPR = 0
for epoch in range(args.ep):
    print("=" * 50)
    print(f"Epoch {epoch+1}/{args.ep}")
    model.train()
    train(drug_loader_train, cline_loader_train, glo_loader_train, label_train)
    AUC, AUPR, Y_true, Y_pred = test(
        drug_loader_test, cline_loader_test, glo_loader_test, label_test
    )
    print("Test metrics:")
    print(f"AUC: {round(AUC, 4)}  AUPR: {round(AUPR, 4)}")
    if AUC > final_AUC:
        final_AUC = AUC
        final_AUPR = AUPR
        print("New best model found! Saving checkpoint...")
        torch.save(model.state_dict(), args.o + "classification_model.pkl")

# print("=" * 50)
# print("Training completed!")
# print(f"Best metrics - AUC: {round(final_AUC, 4)}  AUPR: {round(final_AUPR, 4)}")
# print(f"Total training time: {time.time() - start:.2f} seconds")
# Result.append([final_AUC, final_AUPR])
# # save_prediction_results
# odir = args.o + "classification.txt"
# print(f"Saving results to {odir}")
# np.savetxt(odir, np.array(Result))

# Predict on test set
# Generate batches for final test data
print("Generating final test data batches...")
(
    drug_loader_final_test,
    cline_loader_final_test,
    glo_loader_final_test,
    dc_final_test,
    cc_final_test,
    label_final_test,
) = BatchGenerate(
    test_Independent,
    drug_subfeat,
    cline_subfeat,
    drug_glofeat,
    cline_glofeat,
    drug_compo_elem,
    bs=batch_sizes,
)

# Load the best model
print("Loading best model for final prediction...")
best_model = SubCDR(
    SubEncoder(in_drug=drug_dim, in_cline=8, out=82),
    GraphEncoder(in_channels=32, out_channels=16),
    GloEncoder(in_channels=glo_dim, out_channels=128),
    Decoder(in_channels=160),
).to(device)
best_model.load_state_dict(torch.load(args.o + "classification_model.pkl"))
best_model.eval()

# Make predictions and evaluate on test data
print("Performing final test prediction...")
final_test_AUC, final_test_AUPR, final_test_Y_true, final_test_Y_pred = test(
    drug_loader_final_test,
    cline_loader_final_test,
    glo_loader_final_test,
    label_final_test,
)

print("=" * 50)
print("Final Test Results:")
print(f"Test AUC: {round(final_test_AUC, 4)}")
print(f"Test AUPR: {round(final_test_AUPR, 4)}")

final_test_Y_pred = np.array(final_test_Y_pred) > 0.5

accuracy = accuracy_score(final_test_Y_true, final_test_Y_pred)
precision = precision_score(final_test_Y_true, final_test_Y_pred)
recall = recall_score(final_test_Y_true, final_test_Y_pred)
f1 = f1_score(final_test_Y_true, final_test_Y_pred)

print(f"Accuracy: {round(accuracy, 4)}")
print(f"Precision: {round(precision, 4)}")
print(f"Recall: {round(recall, 4)}")
print(f"F1 Score: {round(f1, 4)}")

test_results_path = args.o + "test_results.csv"
file_exists = os.path.isfile(test_results_path)

# Write header if file doesn't exist
if not file_exists:
    with open(test_results_path, "w") as f:
        f.write("ACC,Precision,Recall,F1,AUC,AUPR\n")

# Append results to file
with open(test_results_path, "a") as f:
    f.write(
        f"{accuracy},{precision},{recall},{f1},{final_test_AUC},{final_test_AUPR}\n"
    )

print(f"Saving test results to {test_results_path}")
