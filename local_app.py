import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pycaret.classification import predict_model

# Cargar el modelo preentrenado

with open('best_model.pkl', 'rb') as model_file:
    dt2 = pickle.load(model_file)

# Cargar los datos de prueba
prueba = pd.read_csv("prueba_APP.csv",header = 0,sep=";",decimal=",")

# Título de la API
st.title("API de Predicción Precio")

# Entradas del usuario para los selectbox en el orden de la imagen
Address = st.selectbox("Address", ['Munich', 'Ausburgo', 'Berlin', 'Frankfurt'], index=0)
dominio = st.selectbox("dominio", ['yahoo', 'Otro', 'gmail', 'hotmail'], index=0)
Tec = st.selectbox("Tec", ['PC', 'Smartphone', 'Iphone', 'Portatil'], index=0)
Avg_Session_Length = st.text_input("Avg Session Length", value="33.946241")
Time_on_App = st.text_input("Time on App", value="10.983977")
Time_on_Website = st.text_input("Time on Website", value="37.951489")
Length_of_Membership = st.text_input("Length of Membership", value="3.050713")

# Convertir los valores de texto a números si es posible
if st.button("Calcular"):
    try:

        # Crear el dataframe a partir de los inputs del usuario
        user = pd.DataFrame({
            'x0':['amarismartinezd@gmail.com'],'x1':[Address],'x2':[dominio],'x3': [Tec],
            'x4': [Avg_Session_Length], 'x5': [Time_on_App], 'x6': [Time_on_Website], 'x7':[Length_of_Membership], 'x8':[0]
        })

        # Asegurar que las columnas coincidan con las del dataset de prueba
        user.columns = prueba.columns

        # Concatenar los datos del usuario con los datos de prueba
        prueba2 = pd.concat([user,prueba],axis = 0)
        prueba2.index = range(prueba2.shape[0])

        # Hacer predicciones
        predictions = predict_model(dt2, data=prueba2)

        st.write(f'La predicción es: {predictions.iloc[0]["prediction_label"]}')

    except ValueError:
        st.error("Por favor, ingrese valores numéricos válidos en los campos correspondientes.")

# Botón para reiniciar la consulta
if st.button("Reiniciar"):
    st.experimental_rerun()