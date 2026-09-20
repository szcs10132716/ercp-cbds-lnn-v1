# -*- coding: utf-8 -*-
"""
train.py: Pipeline script to train the Liquid Neural Network on CBDS recurrence data.
"""

import os, sys, copy, random, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTENC
import joblib

try:
    from src.ltc_model import LTCCell
except ImportError:
    from ltc_model import LTCCell

FEATURES = ['PAD', 'Stone_count', 'Stone_diameter', 'CBD_diameter', 'EST', 'CBDA']
CAT_IDX  = [0, 1, 4, 5]

def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

def train(data_path, out_dir="model", hidden=112, steps=5, epochs=200, lr=0.00223, weight_decay=2.19e-5):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(42)
    
    print(f"Loading dataset from: {data_path}")
    if data_path.endswith(".xlsx"):
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path)
        
    X = df[FEATURES].values.astype(float)
    y = df['Label'].values.astype(int)
    
    print(f"Cohort size: {len(X)}, Events: {y.sum()} ({y.mean():.2%})")
    
    # 7:3 Partition
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.30, random_state=3)
    X_fit, X_es, y_fit, y_es = train_test_split(X_tr, y_tr, test_size=0.15, random_state=42, stratify=y_tr)
    
    # SMOTENC on training cohort
    sm = SMOTENC(categorical_features=CAT_IDX, random_state=0, k_neighbors=5)
    Xfr, yfr = sm.fit_resample(X_fit, y_fit)
    
    # Scaler
    scaler = MinMaxScaler().fit(Xfr)
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
    
    X_fit_sc = scaler.transform(Xfr)
    X_es_sc  = scaler.transform(X_es)
    X_val_sc = scaler.transform(X_val)
    
    # Initialize Model
    model = LTCCell(in_dim=6, hidden=hidden, steps=steps)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    
    dl = DataLoader(TensorDataset(torch.tensor(X_fit_sc, dtype=torch.float32),
                                  torch.tensor(yfr, dtype=torch.long)), batch_size=16, shuffle=True)
    Xe, ye = torch.tensor(X_es_sc, dtype=torch.float32), torch.tensor(y_es, dtype=torch.long)
    
    best_loss, best_state, wait = np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xe), ye).item()
        if vl < best_loss - 1e-5:
            best_loss, best_state, wait = vl, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= 10:
                break
                
    if best_state:
        model.load_state_dict(best_state)
        
    weights_path = os.path.join(out_dir, "lnn_weights.pt")
    torch.save(model.state_dict(), weights_path)
    print(f"Model saved to: {weights_path}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        p_val = F.softmax(model(torch.tensor(X_val_sc, dtype=torch.float32)), dim=-1)[:, 1].numpy()
        
    val_auc = roc_auc_score(y_val, p_val)
    print(f"Validation AUROC: {val_auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to input training data")
    parser.add_argument("--out", type=str, default="model", help="Output directory")
    args = parser.parse_args()
    train(args.data, out_dir=args.out)
