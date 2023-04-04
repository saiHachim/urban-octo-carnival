

#sujet : Identifier et prédire les fraudes financières à partir
 #       des données de transactions historiques.


#Étape 1 : Importer les bibliothèques nécessaires et charger les données

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler ,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense


# In[2]:


# Charger les données
data = pd.read_csv('/Users/hachim/Downloads/Iris.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values



data.head()




data.info()



#Étape 3 : Préparer les données pour la classification

# Supprimer la colonne "Id"
data = data.drop('Id', axis=1)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1].values, data.iloc[:, -1].values, test_size=0.2, random_state=42)

# Encoder les étiquettes de classe
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# Normaliser les données d'entrée
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[7]:


#Étape 4 : Construire le modèle de classification

# Créer le modèle
model = Sequential()
model.add(Dense(10, input_shape=(4,), activation='relu'))
model.add(Dense(3, activation='softmax'))


# In[8]:


#Étape 5 : Entraîner le modèle de classification

# Compiler le modèle
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# Entraîner le modèle
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)



#Étape 6 : Évaluer les performances du modèle de classification

# Charger les données
data = pd.read_csv('/Users/hachim/Downloads/Iris.csv')


# In[11]:


# Prétraitement des données

le = LabelEncoder()
data['Species'] = le.fit_transform(data['Species'])
X = data.drop(['Id', 'Species'], axis=1)
y = data['Species']
X = (X - X.mean()) / X.std() # Standardisation


# In[12]:


# Séparation en ensemble d'entraînement et ensemble de test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[13]:


# Prédire les classes pour les données de test
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)




# Calculer l'exactitude et la matrice de confusion

accuracy = accuracy_score(y_test, y_pred)

confusion_mtx = confusion_matrix(y_test, y_pred)


# In[ ]:





# In[15]:


# Afficher les résultats
print('Accuracy:', accuracy)
print('Confusion matrix:\n', confusion_mtx)




