import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# input
attack_method = "fgsm"     # "fgsm", "pgd"
epsilon = 0              # epsilon value (use 0 for clean)
model_type = "robust_cnn"  # "simple_mlp", "simple_cnn", "robust_cnn"
epoch = 8
trial = 3

if epsilon == 0 or attack_method.lower() == "clean":
    folder_name = "clean"
else:
    folder_name = f"{attack_method.lower()}_{epsilon}"

filename = f"{model_type}_ep{epoch:02d}_t{trial}_cm.npy"

base_path = os.path.join("visuals", "confusion_matrices", folder_name)
full_path = os.path.join(base_path, filename)

if os.path.exists(full_path):
    print(f"Loading matrix from: {full_path}")
    cm = np.load(full_path)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    title_str = f"Confusion Matrix: {model_type.upper()}\nMethod: {folder_name} | Epoch: {epoch} | Trial: {trial}"
    plt.title(title_str)
    
    plt.show()
else:
    print(f"Error: File not found at {full_path}")
