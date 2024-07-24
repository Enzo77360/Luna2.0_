import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Concatenate

# Nombre de caractéristiques dans la séquence
n_features = 4097

# Nombre de caractéristiques dans le paramètre conditionnel
cond_features = 5

# Fonction d'activation des couches Dense
a_func = 'tanh'

# Nom du modèle
model_name = 'Dense_Model.h5'

# Séquence d'entrée
i = Input(shape=[n_features], name='sequential_input')

# Paramètres conditionnels
c = Input(shape=[cond_features], name='conditional_parameter')

# Combiner les entrées séquentielles et les paramètres conditionnels
combined_input = Concatenate()([i, c])

# Définir le modèle PINN
def create_PINN():
    model = Sequential()

    # Couche d'entrée combinée
    model.add(Input(shape=(n_features + cond_features,)))

    # 4 couches cachées avec 100 neurones chacune et activation tanh
    for _ in range(4):
        model.add(Dense(100, activation=a_func))

    # Couche de sortie
    model.add(Dense(n_features))

    return model

# Créer le modèle
model = create_PINN()

# Compiler le modèle
model.compile()

# Afficher le résumé du modèle
model.summary()

# Enregistrer le modèle
model.save(model_name)
