import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import json

# Carga los datos de entrenamiento y prueba utilizando pandas
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Vectoriza los datos de entrenamiento utilizando CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_df['text'])
y_train = train_df['label']

# Crea un objeto MultinomialNB y lo entrena con los datos de entrenamiento vectorizados
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Vectoriza los datos de prueba y utiliza el modelo para predecir las etiquetas
X_test = vectorizer.transform(test_df['text'])
y_pred = clf.predict(X_test)

# Genera un archivo json con los resultados de las predicciones en el formato especificado
predictions = {"target": {str(idx):int(label) for idx, label in zip(test_df['test_idx'], y_pred)}}
with open('predictions.json', 'w') as f:
    json.dump(predictions, f)

# Imprime en pantalla el número de reseñas de entrenamiento y prueba
print(f'Número de reseñas de entrenamiento: {len(train_df)}')
print(f'Número de reseñas de prueba: {len(test_df)}')

# Imprime en pantalla el número de reseñas positivas y negativas en los datos de entrenamiento
num_positive = sum(train_df['label'])
num_negative = len(train_df) - num_positive
print(f'Número de reseñas positivas en los datos de entrenamiento: {num_positive}')
print(f'Número de reseñas negativas en los datos de entrenamiento: {num_negative}')

# Imprime en pantalla el número de reseñas positivas y negativas predichas en los datos de prueba
num_predicted_positive = sum(y_pred)
num_predicted_negative = len(y_pred) - num_predicted_positive
print(f'Número de reseñas positivas predichas en los datos de prueba: {num_predicted_positive}')
print(f'Número de reseñas negativas predichas en los datos de prueba: {num_predicted_negative}')
