#!/usr/bin/env python3
"""
use_model.py - Script to load the trained model and run inference.
"""
import os
import sys
import numpy as np
import torch
from torchvision import datasets, transforms
from model import SmallCNN

MODEL_PATH = "trained_model.npy"
DATA_DIR = os.path.join("data", "val")

def load_trained_model(path):
    if not os.path.exists(path):
        print(f"[!] Model file {path} not found. Please run the training demo first.")
        return None
    
    # Load flattened weights
    print(f"[*] Loading weights from {path}...")
    flat_weights = np.load(path)
    
    # Instantiate model
    model = SmallCNN()
    
    # Reshape and load weights into model
    idx = 0
    with torch.no_grad():
        for p in model.parameters():
            s = list(p.size())
            count = int(np.prod(s))
            # Reshape slice of flat array to match parameter shape
            arr = flat_weights[idx:idx+count].reshape(s)
            p.copy_(torch.from_numpy(arr))
            idx += count
            
    model.eval()
    print("[+] Model loaded successfully.")
    return model

def run_inference(model):
    if not os.path.exists(DATA_DIR):
        print(f"[!] Validation data directory {DATA_DIR} not found.")
        return

    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ])
    
    print(f"[*] Loading validation data from {DATA_DIR}...")
    try:
        ds = datasets.ImageFolder(DATA_DIR, transform=transform)
    except Exception as e:
        print(f"[!] Error loading data: {e}")
        return

    # Pick 5 random samples
    indices = np.random.choice(len(ds), 5, replace=False)
    
    print("\n--- Inference Results ---")
    correct_count = 0
    
    # Get class names if available
    classes = ds.classes
    print(f"Classes: {classes}")

    with torch.no_grad():
        for i in indices:
            img, label = ds[i]
            # Add batch dimension
            img_batch = img.unsqueeze(0)
            
            output = model(img_batch)
            pred_idx = output.argmax(dim=1).item()
            
            status = "✅" if pred_idx == label else "❌"
            if pred_idx == label: correct_count += 1
            
            print(f"Sample {i}: True='{classes[label]}' vs Pred='{classes[pred_idx]}' {status}")
            
    print(f"\nAccuracy on random batch: {correct_count}/5 ({correct_count/5*100:.0f}%)")

if __name__ == "__main__":
    model = load_trained_model(MODEL_PATH)
    if model:
        run_inference(model)
