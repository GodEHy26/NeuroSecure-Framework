import os 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation 

# --- 1. SETTINGS --- 
model = "simple_mlp" 
epoch = 10 
epsilons = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5] 
attack = "fgsm" 

# Setup the figure (Normal White Style) 
plt.style.use('default') 
fig, ax = plt.subplots(figsize=(8, 8)) 

def update(eps): 
    ax.clear() 
    folder = "clean" if eps == 0.0 else f"{attack}_{eps}" 
     
    # Load and average trials 
    cms = [] 
    for t in [1, 2, 3]: 
        path = f"visuals/confusion_matrices/{folder}/{model}_ep{epoch:02d}_t{t}_cm.npy" 
        if os.path.exists(path): 
            cms.append(np.load(path)) 
     
    if cms: 
        avg_cm = np.mean(cms, axis=0) 
        norm_cm = avg_cm / avg_cm.sum(axis=1)[:, np.newaxis] 
         
        # Plotting with Blues
        sns.heatmap(norm_cm, annot=False, cmap='Blues', cbar=False, square=True, ax=ax, vmin=0, vmax=1) 
        ax.set_title(f"Model Logic at ε = {eps}", fontsize=20, color='black', pad=20) 
        ax.set_xlabel("Predicted Label", color='black') 
        ax.set_ylabel("True Label", color='black') 
    else: 
        ax.text(0.5, 0.5, f"Missing Data: {eps}", ha='center') 

# Create Animation 
ani = FuncAnimation(fig, update, frames=epsilons, repeat=True, interval=500) 

# Save as GIF 
output_name = f"{model}_.gif" 
ani.save(output_name, writer='pillow', fps=2) 
print(f"✅ Animation saved as {output_name}") 
plt.show()