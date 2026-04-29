import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

df = pd.read_csv('results/adversarial_attack_results.csv')
model_label = 'simple_cnn' # input
df_model = df[df['Model_Type'] == model_label]
pivot = df_model.pivot_table(index='Epochs', columns='Epsilon', values='Adv_Accuracy', aggfunc='mean')

eps_dense = np.linspace(pivot.columns.min(), pivot.columns.max(), 50)
ep_dense = np.linspace(pivot.index.min(), pivot.index.max(), 50)
X, Y = np.meshgrid(eps_dense, ep_dense)

points = [(eps, ep) for ep in pivot.index for eps in pivot.columns]
values = pivot.values.flatten()
Z = griddata(points, values, (X, Y), method='cubic')

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9, antialiased=True)

ax.view_init(elev=30, azim=-60)

ax.set_title(f'Smoothened Robustness Surface: {model_label.upper()}')
ax.set_xlabel('Epsilon (Noise)')
ax.set_ylabel('Epochs')
ax.set_zlabel('Accuracy %')
plt.savefig(f'3d_smooth_{model_label}.png')
plt.show()
