# -*- coding: utf-8 -*-
"""
predict.py: Clinical inference and risk stratification pipeline for CBDS recurrence after ERCP.
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import joblib

# Import LTCCell architecture
try:
    from src.ltc_model import LTCCell
except ImportError:
    from ltc_model import LTCCell

class LNNPredictor:
    """
    End-to-end inference wrapper combining IterativeImputer, MinMaxScaler, and LNN weights.
    """
    def __init__(self, model_dir=None):
        if model_dir is None:
            # Assume default model/ directory relative to repository root
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(base_dir, "model")
            
        self.model_dir = model_dir
        self.config_path = os.path.join(model_dir, "config.json")
        self.weights_path = os.path.join(model_dir, "lnn_weights.pt")
        self.scaler_path = os.path.join(model_dir, "scaler.joblib")
        self.imputer_path = os.path.join(model_dir, "imputer.joblib")
        
        self._load_artifacts()

    def _load_artifacts(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            
        self.features = self.config["features"]
        self.threshold = float(self.config["decision_threshold"])
        arch = self.config["architecture"]
        
        # Instantiate model
        self.model = LTCCell(
            in_dim=arch["in_dim"],
            hidden=arch["hidden_dim"],
            steps=arch["steps"],
            dt=arch["dt"]
        )
        self.model.load_state_dict(torch.load(self.weights_path, map_location="cpu", weights_only=True))
        self.model.eval()
        
        # Preprocessors
        self.scaler = joblib.load(self.scaler_path)
        self.imputer = joblib.load(self.imputer_path) if os.path.exists(self.imputer_path) else None

    def predict_dataframe(self, df_input):
        """
        Takes a pandas DataFrame containing the 6 required features and returns predictions.
        
        Parameters:
            df_input (pd.DataFrame): Must contain ['PAD', 'Stone_count', 'Stone_diameter', 
                                                  'CBD_diameter', 'EST', 'CBDA']
                                                  
        Returns:
            pd.DataFrame: Original dataframe appended with:
                          - Predicted_Recurrence_Probability
                          - Risk_Stratification
                          - Binary_Prediction_Threshold_0.198
        """
        missing_cols = [c for c in self.features if c not in df_input.columns]
        if missing_cols:
            raise ValueError(f"Input DataFrame is missing required predictor columns: {missing_cols}")
            
        X_raw = df_input[self.features].values.astype(float)
        
        # Impute if missing values are present
        if np.isnan(X_raw).any():
            if self.imputer is not None:
                X_raw = self.imputer.transform(X_raw)
            else:
                raise ValueError("Input contains missing values but imputer is unavailable.")
                
        # Scale
        X_scaled = self.scaler.transform(X_raw)
        
        # Inference
        with torch.no_grad():
            logits = self.model(torch.tensor(X_scaled, dtype=torch.float32))
            probs = F.softmax(logits, dim=-1)[:, 1].numpy()
            
        df_res = df_input.copy()
        df_res["Predicted_Recurrence_Probability"] = np.round(probs, 4)
        df_res["Binary_Prediction_Threshold_0.198"] = (probs >= self.threshold).astype(int)
        df_res["Risk_Stratification"] = np.where(
            probs >= self.threshold, 
            f"Higher Risk (>= {self.threshold})", 
            f"Lower Risk (< {self.threshold})"
        )
        return df_res

    def predict_single(self, pad, stone_count, stone_diameter, cbd_diameter, est, cbda):
        """
        Predict for a single individual.
        """
        df_single = pd.DataFrame([{
            "PAD": pad,
            "Stone_count": stone_count,
            "Stone_diameter": stone_diameter,
            "CBD_diameter": cbd_diameter,
            "EST": est,
            "CBDA": cbda
        }])
        res = self.predict_dataframe(df_single)
        prob = res["Predicted_Recurrence_Probability"].iloc[0]
        strat = res["Risk_Stratification"].iloc[0]
        return prob, strat

def main():
    parser = argparse.ArgumentParser(description="CBDS Recurrence Prediction using Liquid Neural Network (LNN)")
    parser.add_argument("--input", type=str, default=None, help="Path to input CSV file")
    parser.add_argument("--output", type=str, default=None, help="Path to output CSV file")
    parser.add_argument("--pad", type=int, default=None, help="Periampullary diverticulum (0=No, 1=Yes)")
    parser.add_argument("--stone_count", type=int, default=None, help="Stone count (1=1, 2=2, 3=>=3)")
    parser.add_argument("--stone_diameter", type=float, default=None, help="Stone diameter (mm)")
    parser.add_argument("--cbd_diameter", type=float, default=None, help="CBD diameter (mm)")
    parser.add_argument("--est", type=int, default=None, help="EST (0=No, 1=Yes)")
    parser.add_argument("--cbda", type=int, default=None, help="CBDA (0=>145 deg, 1=<=145 deg)")
    
    args = parser.parse_args()
    predictor = LNNPredictor()
    
    if args.input:
        print(f"Reading input file: {args.input}")
        df_in = pd.read_csv(args.input)
        df_out = predictor.predict_dataframe(df_in)
        out_path = args.output if args.output else "predictions.csv"
        df_out.to_csv(out_path, index=False)
        print(f"Saved predictions to: {out_path}")
        print(df_out[["Predicted_Recurrence_Probability", "Risk_Stratification"]].head())
        
    elif args.pad is not None:
        prob, strat = predictor.predict_single(
            pad=args.pad,
            stone_count=args.stone_count,
            stone_diameter=args.stone_diameter,
            cbd_diameter=args.cbd_diameter,
            est=args.est,
            cbda=args.cbda
        )
        print("\n=======================================================")
        print("          LNN CBDS Recurrence Risk Assessment")
        print("=======================================================")
        print(f"Predicted Recurrence Probability: {prob:.2%}")
        print(f"Decision Cutoff Threshold       : {predictor.threshold:.3f}")
        print(f"Risk Stratification Category    : {strat}")
        print("=======================================================\n")
    else:
        print("Please provide --input <file.csv> or individual clinical arguments. Run with --help for details.")

if __name__ == "__main__":
    main()
