import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv1D, Concatenate, Flatten, Layer

# Nombre de caractéristiques dans la séquence
n_features = 4097

# Nombre de caractéristiques dans le paramètre conditionnel
cond_features = 5

# Fonction d'activation des couches Dense
a_func = 'tanh'

# Nom du modèle
model_name = 'Conv_Model.h5'

# Définir les entrées
seq_input = Input(shape=(n_features,), name='sequential_input')
cond_input = Input(shape=(cond_features,), name='conditional_parameter')

# Concaténer les caractéristiques séquentielles avec les paramètres conditionnels
combined_input = Concatenate()([seq_input, cond_input])
print(f"Shape after Concatenate: {combined_input.shape}")

# Créer une couche personnalisée pour ajouter une dimension
class ExpandDimsLayer(Layer):
    def call(self, inputs):
        return tf.expand_dims(inputs, axis=1)

# Ajouter une dimension pour les convolutions
expanded_input = ExpandDimsLayer()(combined_input)
print(f"Shape after ExpandDimsLayer: {expanded_input.shape}")

# Créer le modèle
x = Conv1D(filters=128, kernel_size=1, activation='relu')(expanded_input)
x = Conv1D(filters=128, kernel_size=1, activation='relu')(x)
x = Conv1D(filters=128, kernel_size=1, activation='relu')(x)
x = Flatten()(x)
x = Dense(100, activation=a_func)(x)
x = Dense(100, activation=a_func)(x)

# Couche de sortie
output = Dense(n_features)(x)

# Définir le modèle
model = Model(inputs=[seq_input, cond_input], outputs=output)

# Compiler le modèle
model.compile()

# Afficher le résumé du modèle
model.summary()

# Enregistrer le modèle
model.save(model_name)
