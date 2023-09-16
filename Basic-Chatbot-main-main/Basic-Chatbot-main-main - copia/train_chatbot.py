# Importa las bibliotecas necesarias
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import random
import json
import pickle
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer  # Importa el lematizador de NLTK

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Inicializa el lematizador de palabras
lemmatizer = WordNetLemmatizer()

# Listas para almacenar palabras, clases e información de documentos
words = []
classes = []
documents = []
ignore_letters = ['!', '?', '¿', ',', '.']  # Caracteres a ignorar
intents_file = open('intents.json').read()  # Carga el archivo JSON con las intenciones
intents = json.loads(intents_file)

# Procesa las intenciones y patrones
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokeniza cada palabra en el patrón
        word = nltk.word_tokenize(pattern)
        words.extend(word)  # Agrega las palabras a la lista de palabras
        # Agrega documentos al corpus con la intención correspondiente
        documents.append((word, intent['tag']))
        # Agrega la intención a la lista de clases si no existe previamente
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print(documents)

# Lematiza las palabras, las convierte a minúsculas y elimina duplicados
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))  # Ordena las palabras únicas
classes = sorted(list(set(classes)))  # Ordena las clases únicas

# Imprime estadísticas
print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

# Guarda las listas de palabras y clases en archivos binarios
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepara los datos para el entrenamiento de la red neuronal

# Crea una lista para almacenar los datos de entrenamiento
training = []
# Crea un array vacío para las salidas
output_empty = [0] * len(classes)

# Genera el conjunto de entrenamiento en formato "bag of words" para cada patrón
for doc in documents:
    bag = []  # Inicializa el "bag of words" para cada patrón
    pattern_words = doc[0]
    # Lematiza cada palabra y conviértela a minúsculas
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    # Crea el "bag of words" con 1 si la palabra coincide con el patrón actual
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    # Genera la salida como '0' para cada etiqueta y '1' para la etiqueta actual (para cada patrón)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Mezcla las características de entrenamiento y conviértelas en un arreglo numpy
random.shuffle(training)
training = np.array(training, dtype="object")
print(training)

# Crea listas de entrenamiento y prueba (X - patrones, Y - intenciones)
train_x = list(training[:, 0])
train_y = list(training[:, 1])
print("Training data created")

# Creación del modelo de la red neuronal

# Crea un modelo secuencial
model = Sequential()

# Agrega una capa densa con 128 neuronas de entrada, activación ReLU y dropout
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))

# Agrega una segunda capa densa con 64 neuronas, activación ReLU y dropout
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Agrega la capa de salida con un número de neuronas igual al número de intenciones y activación softmax
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compila el modelo con un optimizador SGD (descenso de gradiente estocástico)
sgd = tf.keras.optimizers.legacy.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrena el modelo y guarda el historial de entrenamiento
hist = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)

# Guarda el modelo entrenado en un archivo
model.save('chatbot_model.h5', hist)

print("Model created")
