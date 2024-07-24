import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from ConvModel.CreateConvModel import ExpandDimsLayer
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope

# Chemin vers le répertoire contenant les fichiers CSV
directory = "P:/FewData"

# Liste pour stocker les données de chaque fichier CSV
data_list = []

# Parcourir les fichiers CSV dans le répertoire
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        file_path = os.path.join(directory, filename)
        data = pd.read_csv(file_path)
        data_list.append(data)

# Initialiser les listes pour stocker les données d'entraînement et de validation
train_x = []
train_y = []
train_c = []

# Préparer les données d'entraînement et de validation
for data in data_list:
    # Extraire les conditions (constantes pour chaque fichier CSV)
    conditions = data[['pressure', 'trange', 'length_HCF', 'energy']].iloc[0].values

    # Préparer les données Iw_real en excluant les colonnes inutiles
    Iw = data.iloc[:, 6:207].values  # Colonnes de la 7e à la dernière colonne pour Iw_real

    # Transposer si nécessaire pour obtenir la forme (2049, num_observations)
    if Iw.shape[0] < Iw.shape[1]:
        Iw = Iw.T

    # Normaliser les données par le maximum de Iw
    max_Iw = np.max(Iw)
    Iw_normalized = Iw / max_Iw

    # Convertir les valeurs normalisées en dB
    Iw_dB = 10 * np.log10(Iw_normalized)  # Ajout d'une petite valeur pour éviter le log(0)

    # Normaliser les valeurs en dB entre 0 et 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    Iw_dB_normalized = scaler.fit_transform(Iw_dB.T).T

    num_observations = Iw.shape[1]

    x_seq_real = Iw_dB_normalized[:, 0]  # Spectre d'entrée

    # Générer les séquences pour l'entraînement
    for j in range(num_observations):
        x_seq_real = Iw_dB_normalized[:, 0]  # Utiliser la première colonne comme spectre d'entrée
        y_seq_real = Iw_dB_normalized[:, j]  # Valeur vraie pour y_train_real

        condition_with_index = np.concatenate([conditions, [j]])  # Ajouter l'indice j aux conditions

        train_x.append(x_seq_real)
        train_y.append(y_seq_real)
        train_c.append(condition_with_index)

# Convertir les listes en tableaux numpy
train_x = np.array(train_x)
train_y = np.array(train_y)
train_c = np.array(train_c)

# Afficher quelques échantillons des données d'entraînement et de validation
print("Exemples de données d'entraînement:")
print("train_x[0]:", train_x[0])
print("train_y[0]:", train_y[0])
print("train_c[0]:", train_c[0])

print(f"\nShape of train_x: {train_x.shape}")
print(f"Shape of train_y: {train_y.shape}")
print(f"Shape of train_c: {train_c.shape}")

# Définir la fonction RMSE
def rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square((y_true - y_pred))))


# Charger le modèle avec la couche personnalisée
model_path = 'Conv_Model.h5'
with custom_object_scope({'ExpandDimsLayer': ExpandDimsLayer}):
    model = load_model(model_path)



# Compiler le modèle avec une fonction de perte et un optimiseur appropriés
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=rmse, metrics=[rmse])

# Définir l'arrêt anticipé avec une patience de 5 epochs
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entraîner le modèle en utilisant 20% des données pour la validation
history = model.fit(
    x=[train_x, train_c], y=train_y,
    batch_size=256,
    validation_split=0.2,  # Utiliser 20% des données pour la validation
    epochs=100,  # Vous pouvez ajuster ce nombre d'epochs selon vos besoins
    verbose=1,
    callbacks=[early_stopping]  # Ajouter l'arrêt anticipé
)

# Plot training losses with subplots
plt.figure(figsize=(16, 6))

# Total Loss subplot
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Total Loss')
plt.plot(history.history['val_loss'], label='Validation Total Loss')
plt.title('Total Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# MAE Loss subplot
plt.subplot(1, 2, 2)
plt.plot(history.history['rmse'], label='Train RMSE')
plt.plot(history.history['val_rmse'], label='Validation RMSE')
plt.title('Root Mean Squared Error')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# test_x et test_c avec les 201 dernieres observations de train_x et train_c
# Plot des prédictions avant la sauvegarde du modèle
predictions = model.predict([train_x, train_c])  # test_x et test _c
print(predictions)
print(predictions.shape)

# Sélectionner quelques exemples pour l'affichage
num_examples = 3
indices = np.random.choice(len(train_x), num_examples, replace=False)

# Afficher les prédictions et les valeurs réelles
plt.figure(figsize=(16, 10))
for i, idx in enumerate(indices):
    # Prédiction
    pred = predictions[idx]

    # Valeur réelle
    true_value = train_y[idx]

    # Plot
    plt.subplot(num_examples, 1, i + 1)
    plt.plot(pred, label='Prediction')
    plt.plot(true_value, label='True Value')
    plt.title(f'Exemple {i+1}')
    plt.xlabel('Time Step')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Sauvegarder le modèle entraîné
model.save('Trained_Conv_2000.h5')