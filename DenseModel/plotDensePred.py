import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square((y_true - y_pred) / y_true)))

# Charger le modèle existant
clean_model_name = 'P:/Codes Python/Luna2.0_/Trained_Dense.h5'
model = tf.keras.models.load_model(clean_model_name, custom_objects={'rmse': rmse})

# Chemin vers le fichier CSV contenant le nouveau dataset
csv_file = "P:/FewData/output_data_core250μm_gasN2_pressure3.0bar_time30fs_length2m_energy150μJ.csv"

# Charger les données CSV
data = pd.read_csv(csv_file)

# Extraire les conditions (constantes pour chaque fichier CSV)
conditions = data[['pressure', 'trange', 'length_HCF', 'energy']].iloc[0].values
print("Conditions:", conditions)

# Préparer les données Iw en excluant les colonnes inutiles
Iw = data.iloc[:, 6:207].values
print("Forme initiale de Iw:", Iw.shape)

# Transposer si nécessaire pour obtenir la forme (2049, num_observations)
if Iw.shape[0] < Iw.shape[1]:
    Iw = Iw.T
print("Forme de Iw après transposition (si nécessaire):", Iw.shape)

# Normaliser Iw par sa valeur maximale
Iw = Iw / np.max(Iw)
# Convertir les valeurs en dB
Iw_dB = 10 * np.log10(Iw + 1e-9)  # Ajout d'une petite valeur pour éviter le log(0)

# Normaliser les valeurs en dB entre 0 et 1
scaler = MinMaxScaler(feature_range=(0, 1))
Iw_dB_normalized = scaler.fit_transform(Iw_dB.T).T

num_observations = Iw_dB_normalized.shape[1]
print("Nombre d'observations:", num_observations)

# Utiliser la première colonne comme spectre d'entrée
x_seq_real = Iw_dB_normalized[:, 0]

# Préparer les conditions avec l'indice de l'observation
conditions_expanded_list = []
for j in range(num_observations):
    condition_with_index = np.concatenate([conditions, [j]])  # Ajouter l'indice j aux conditions
    conditions_expanded_list.append(condition_with_index)

conditions_expanded_array = np.array(conditions_expanded_list)

# Répéter le spectre d'entrée pour chaque observation
x_seq_real_repeated = np.tile(x_seq_real, (num_observations, 1))

# Concaténer les entrées séquentielles et les conditions
x_c_test = np.hstack((x_seq_real_repeated, conditions_expanded_array))
print("Taille de x_c_test:", x_c_test.shape)

# Prédire toutes les valeurs en une seule fois
predictions = model.predict(x_c_test)
print("Forme des prédictions:", predictions.shape)

# Convertir les prédictions en tableau numpy
predictions = predictions.squeeze()
print("Forme des prédictions après suppression des dimensions inutiles:", predictions.shape)

# Assurez-vous que la forme des prédictions est correcte pour la transformation inverse
if predictions.shape != Iw_dB_normalized.shape:
    # Transposez les prédictions si nécessaire pour faire correspondre les dimensions
    predictions = predictions.T

# Dénormaliser les prédictions pour obtenir les valeurs en dB
predictions_dB = scaler.inverse_transform(predictions.T).T

# Créer des sous-graphes pour comparer les vraies valeurs et les prédictions
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

# Carte de chaleur des vraies valeurs
im = axes[0].pcolormesh(np.arange(Iw_dB.shape[0]), np.arange(Iw_dB.shape[1]), Iw_dB.T, shading='auto', cmap='plasma', vmin=-60, vmax=0)
fig.colorbar(im, ax=axes[0], label='Vraies valeurs (dB)')
axes[0].set_title('Carte de chaleur des vraies valeurs (échelle logarithmique dB)')
axes[0].set_ylabel('Observations')
axes[0].set_xlabel('Fréquence (PHz)')

# Carte de chaleur des prédictions
im = axes[1].pcolormesh(np.arange(predictions_dB.shape[0]), np.arange(predictions_dB.shape[1]), predictions_dB.T, shading='auto', cmap='plasma', vmin=-60, vmax=0)
fig.colorbar(im, ax=axes[1], label='Prédictions (dB)')
axes[1].set_title('Carte de chaleur des prédictions (échelle logarithmique dB)')
axes[1].set_ylabel('Observations')
axes[1].set_xlabel('Fréquence (PHz)')

plt.tight_layout()
plt.show()

# Créer une nouvelle figure pour les étapes spécifiques
fig2, axes2 = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))

# Étapes spécifiques
steps = [25, 100, 180]
for i, step in enumerate(steps):
    ax = axes2[i]
    # Trace les vraies valeurs
    ax.plot(Iw_dB[:, step], label='Vraies valeurs', color='blue')
    # Trace les prédictions
    ax.plot(predictions_dB[:, step], label='Prédictions', color='red', linestyle='--')
    ax.set_title(f'Étape {step}')
    ax.set_xlabel('Fréquence (PHz)')
    ax.set_ylabel('Valeur (dB)')
    ax.legend()

plt.tight_layout()
plt.show()
