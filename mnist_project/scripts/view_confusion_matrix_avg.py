import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# inputs
attack_method = "fgsm"     # "fgsm", "pgd"
epsilon = 0.15             # epsilon value (use 0 for clean)
model_type = "simple_mlp"  # "simple_mlp", "simple_cnn", "robust_cnn"
epoch = 10

if epsilon == 0 or attack_method.lower() == "clean":
    folder_name = "clean"
else:
    folder_name = f"{attack_method.lower()}_{epsilon}"

base_path = os.path.join("visuals", "confusion_matrices", folder_name)

matrices = []
for t in [1, 2, 3]:
    filename = f"{model_type}_ep{epoch:02d}_t{t}_cm.npy"
    full_path = os.path.join(base_path, filename)
    
    if os.path.exists(full_path):
        print(f"Adding Trial {t} from: {full_path}")
        matrices.append(np.load(full_path))
    else:
        print(f"Warning: Trial {t} not found at {full_path}")

if len(matrices) > 0:
    avg_cm = np.mean(matrices, axis=0)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    title_str = (f"AVERAGE Confusion Matrix (n={len(matrices)})\n"
                 f"Model: {model_type.upper()} | Method: {folder_name} | Epoch: {epoch}")
    plt.title(title_str)
    
    plt.show()
else:
    print(f"Error: File not found at {full_path}")