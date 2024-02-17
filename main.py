import pandas as pd
import numpy as np
import streamlit as st
from datetime import timedelta
from streamlit_echarts import st_echarts
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
import datetime
import tensorflow as tf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import time
import streamlit as st

def main():
    # Configurar el estilo de la página
    st.set_page_config(
        page_title="WattWise",
        page_icon="💡",
        layout="wide",
    )

    # Establecer el color de fondo y el color del texto
    st.markdown(
        """
        <style>
            body {
                background-color: #f4f4f4;
                color: #333333;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Título con color y emoticono
    st.title("WattWise - Comprenda su consumo Eléctrico 🌐💡")
    
    df = pd.read_csv("electrodatos.csv")

    def explain_visualization(prompt, image):
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids
        image_tensor = model.image_preprocess(image)

        #Generate the answer
        output_ids = model.generate(
            input_ids,
            max_new_tokens=200,
            images=image_tensor,
            use_cache=False,
            temperature=0.1)[0]
        return tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()

    # Enlaces a las subpáginas
    options = ["Comprenda cómo consume Energía", "Ahorre Energía y Dinero"]
    selection = st.sidebar.radio("Ir a:", options)

    if selection == "Comprenda cómo consume Energía":
        st.subheader("Comprenda cómo consume Energía")
        # Agrega aquí el contenido de la primera subpágina
        df['Fecha'] = df['Fecha'].apply(lambda x: pd.Timestamp(x))
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['day_of_week'] = df['datetime'].dt.day_name()

        df['Hour'] = df['datetime'].dt.hour

        def asignar_categoria_tiempo(dt):
            hora = dt.hour
            if 6 <= hora < 14:
                return 'Morning'
            elif 14 <= hora < 22:
                return 'Afternoon'
            else:
                return 'Night'
        
        dias_ordenados = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['Horario'] = df['datetime'].apply(asignar_categoria_tiempo)
        df['Fecha'] = df['Fecha'].apply(lambda x: pd.Timestamp(x))
        df['day_of_week'] = df['datetime'].dt.day_name()
        df['Hour'] = df['datetime'].dt.hour
        grouped_data = df.groupby(['day_of_week', 'Horario'])['Consumo'].mean().unstack()
        grouped_data = grouped_data.reindex(dias_ordenados)

        st.title('Consumo por días de la semana y horario')
        options1 = {
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "legend": {
                "data": ["Morning", "Afternoon", "Night"]
            },
            "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
            "xAxis": {"type": "value"},
            "yAxis": {
                "type": "category",
                "data": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            },
            "series": [
                {
                    "name": "Morning",
                    "type": "bar",
                    "stack": "total",
                    "label": {"show": False},
                    "emphasis": {"focus": "series"},
                    "data": grouped_data['Morning'].tolist(),
                },
                {
                    "name": "Afternoon",
                    "type": "bar",
                    "stack": "total",
                    "label": {"show": False},
                    "emphasis": {"focus": "series"},
                    "data": grouped_data['Afternoon'].tolist(),
                },
                {
                    "name": "Night",
                    "type": "bar",
                    "stack": "total",
                    "label": {"show": False},
                    "emphasis": {"focus": "series"},
                    "data": grouped_data['Night'].tolist(),
                },
            ],
        }

        grafica1 = st_echarts(options=options1, height="500px")

        prompt2 = "A chat between an electric company client and an artificial intelligence assistant. The assistant gives helpful, detailed, clear and polite answers to the user's questions, explaining the answer to the user as to a kid. USER: <image>\nBased on this visualization, extract insights and useful conclusions about my weekly electric consumptions in the morning, afternoon and evening. ASSISTANT:"
        image2 = Image.open("images/visualization2.png")
        with st.spinner('Esperando la explicación de WattWise...'):
            text1 = str(explain_visualization(prompt2, image2))
        st.success('Done!')
        st.markdown(text1)

        fecha_inicio_ultimo_mes = max(df['datetime']) - timedelta(days=30)
        consumo_ultimo_mes = df[df['datetime'] >= fecha_inicio_ultimo_mes]['Consumo'].sum()
        fecha_inicio_mes_anterior = fecha_inicio_ultimo_mes - timedelta(days=30)
        consumo_mes_anterior = df[(df['datetime'] >= fecha_inicio_mes_anterior) & (df['datetime'] < fecha_inicio_ultimo_mes)]['Consumo'].sum()
        aumento_porcentual = ((consumo_ultimo_mes - consumo_mes_anterior) / consumo_mes_anterior) * 100

        st.title('Diferencia mensual en el consumo')
        st.metric(label= "Consumo (kWh)", value=round(consumo_ultimo_mes, 3), delta=round(aumento_porcentual, 2),    delta_color="inverse")

        df_prezo = pd.read_csv("PCB.csv")
        df_prezo['PCB'] = (df_prezo['PCB'].str.replace(',', '.').astype(float))/10
        df_prezo = df_prezo.loc[df_prezo['Hour'] != 24]
        df_prezo['datetime'] = pd.to_datetime(df_prezo['Day'] + ' ' + df_prezo['Hour'].astype(str).str.zfill(2), format='mixed', dayfirst=True)
        df_prezo = df_prezo.drop(['Day'], axis=1)
        df_prezo['Horario'] = df_prezo['datetime'].apply(asignar_categoria_tiempo)
        df_prezo['day_of_week'] = df_prezo['datetime'].dt.day_name()

        # Crear la gráfica de abajo con streamlit
        st.title('Precio por días de la semana y horario')
        grafica3 = st.bar_chart(df_prezo.groupby(['day_of_week', 'Horario'])['PCB'].mean().unstack().reindex(dias_ordenados))

        # prompt3 = "A chat between an electric company client and an artificial intelligence assistant. The assistant gives helpful, detailed, clear and polite answers to the user's questions, explaining the answer to the user as to a kid. USER: <image>\nBased on this visualization, extract insights and useful conclusions about the price of Kwh of electricity by day of the week and time of the day. ASSISTANT:"
        # image3 = Image.open("images/visualization3.png")
        # with st.spinner('Esperando la explicación de WattWise...'):
        #     text2 = explain_visualization(prompt3, image3)
        # st.success('Done!')
        # st.markdown(text2)

        st.title('Consumo semanal por horas del día')
        grafica4 = st.bar_chart(df.groupby(['day_of_week', 'Hour'])['Consumo'].mean().unstack().reindex(dias_ordenados))

        # prompt1 = "A chat between an electric company client and an artificial intelligence assistant. The assistant gives helpful, detailed, clear and polite answers to the user's questions, explaining the answer to the user as to a kid. USER: <image>\nBased on this visualization, extract insights and useful conclusions about my weekly electric consumptions in the different hours. Focus on recognizing patterns about the different hours. ASSISTANT:"
        # image1 = Image.open("images/visualization1.png")
        # with st.spinner('Esperando la explicación de WattWise...'):
        #     text3 = explain_visualization(prompt1, image1)
        # st.success('Done!')
        # st.markdown(text3)
        

        df_prezo['Total'] = df_prezo['PCB'] * df['Consumo']

        st.title('Precio pagado por días de la semana y horario')
        grouped_data1 = df_prezo.groupby(['day_of_week', 'Horario'])['Total'].mean().unstack()
        grouped_data1 = grouped_data1.reindex(dias_ordenados)
        options2 = {
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "legend": {
                "data": ["Morning", "Afternoon", "Night"]
            },
            "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
            "xAxis": {"type": "value"},
            "yAxis": {
                "type": "category",
                "data": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            },
            "series": [
                {
                    "name": "Morning",
                    "type": "bar",
                    "stack": "total",
                    "label": {"show": False},
                    "emphasis": {"focus": "series"},
                    "data": grouped_data1['Morning'].tolist(),
                },
                {
                    "name": "Afternoon",
                    "type": "bar",
                    "stack": "total",
                    "label": {"show": False},
                    "emphasis": {"focus": "series"},
                    "data": grouped_data1['Afternoon'].tolist(),
                },
                {
                    "name": "Night",
                    "type": "bar",
                    "stack": "total",
                    "label": {"show": False},
                    "emphasis": {"focus": "series"},
                    "data": grouped_data1['Night'].tolist(),
                },
            ],
        }
        st_echarts(options=options2, height="600px")

        # prompt4 = "A chat between an electric company client and an artificial intelligence assistant. The assistant gives helpful, detailed, clear and polite answers to the user's questions, explaining the answer to the user as to a kid. USER: <image>\nBased on this visualization, extract insights and useful conclusions about the mean price payed by the user in each day of the week. ASSISTANT:"
        # image4 = Image.open("images/visualization4.png")
        # with st.spinner('Esperando la explicación de WattWise...'):
        #     text4 = explain_visualization(prompt4, image4)
        # st.success('Done!')
        # st.markdown(text4)

    elif selection == "Ahorre Energía y Dinero":
        st.subheader("Ahorre Energía y Dinero")
        # Agrega aquí el contenido de la segunda subpágina
        def requeriments_price():
            # Preprocesado
            df = pd.read_csv("PCB.csv")
            df = df.loc[df['Hour'] != 24]
            df['datetime'] = pd.to_datetime(df['Day'] + ' ' + df['Hour'].astype(str).str.zfill(2), format='mixed', dayfirst=True)
            df = df.drop(['Day', 'Hour'], axis=1)
            df['PCB'] = df['PCB'].str.replace(',', '.').astype(float)
            df['day_of_week'] = df['datetime'].dt.day_name()
            df['Hour'] = df['datetime'].dt.hour
            df['month'] = df['datetime'].dt.month_name()
            df = df[['PCB', 'day_of_week', 'Hour', 'month', 'datetime']]

            # Definimos un econder para tranformar las variables categóricas 
            # (dia de la semana y mes)
            enc_ord = OrdinalEncoder()

            # Indicamos la fecha de separación para los conjuntos de train/val y test
            # y eliminamos la columna datetime
            separacion = datetime.datetime(2023, 4, 30, 23, 59)
            df_train_val = df.loc[df['datetime'] < separacion]
            df_train_val = df_train_val.drop(['datetime'], axis=1)

            # Separamos en train y validation
            n = len(df_train_val)
            df_train = df_train_val[:round(n * 0.7)]

            # Normalizamos por media y desviación típica el PCB
            mu = df_train['PCB'].mean()
            std = df_train['PCB'].std()

            enc_ord.fit(df_train[['day_of_week', 'month']])

            return mu, std, enc_ord


        def price_prediction(mu, std, enc_ord, gru_model, df_usuario):
            # PREPROCESADO
            df_usuario = df_usuario.loc[df_usuario['Hour'] != 24]
            df_usuario['datetime'] = pd.to_datetime(df_usuario['Day'] + ' ' + df_usuario['Hour'].astype(str).str.zfill(2), format='mixed', dayfirst=True)
            df_usuario = df_usuario.drop(['Day', 'Hour'], axis=1)
            df_usuario['PCB'] = df_usuario['PCB'].str.replace(',', '.').astype(float)
            df_usuario['day_of_week'] = df_usuario['datetime'].dt.day_name()
            df_usuario['Hour'] = df_usuario['datetime'].dt.hour
            df_usuario['month'] = df_usuario['datetime'].dt.month_name()
            df_usuario = df_usuario[['PCB', 'day_of_week', 'Hour', 'month']]

            # NORMALIZACIÓN
            df_usuario[['day_of_week', 'month']] = enc_ord.transform(df_usuario[['day_of_week', 'month']])
            df_usuario['PCB'] = (df_usuario['PCB'] - mu)/std

            # PREDICCIÓN
            prediccion = gru_model.predict(df_usuario.to_numpy().reshape(1, 168, 4)).flatten()
            valores_usados = df_usuario["PCB"]
            indices_tiempo = np.arange(len(valores_usados) + len(prediccion))
            plt.plot(indices_tiempo[:len(valores_usados)], valores_usados, label='Valores usados',)
            plt.plot(indices_tiempo[len(valores_usados):], prediccion, label='Predicción')
            plt.xlabel('Horas')
            plt.ylabel('Precio')
            plt.legend()
            plt.title('Predicción del precio')
            #plt.show()
            #print(prediccion)
            st.pyplot()
            st.markdown(prediccion*std+mu)

        def requeriments_consumo():
            # PREPROCESADO
            df = pd.read_csv("electrodatos.csv")
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['day_of_week'] = df['datetime'].dt.day_name()
            df['Hour'] = df['datetime'].dt.hour
            df['month'] = df['datetime'].dt.month_name()
            df_consumo = df[['Consumo', 'day_of_week', 'Hour', 'month', 'Código universal de punto de suministro']]

            enc_ord = OrdinalEncoder()

            for i in range(11):
                if i < 6:
                    tmp_df = df_consumo[df_consumo['Código universal de punto de suministro'] == i]
                    enc_ord.fit(tmp_df[['day_of_week', 'month']])

            return enc_ord

        def consumo_prediction(enc_ord, gru_model, df_usuario):
            # PREPROCESADO
            df_usuario['datetime'] = pd.to_datetime(df_usuario['datetime'])
            df_usuario['day_of_week'] = df_usuario['datetime'].dt.day_name()
            df_usuario['Hour'] = df_usuario['datetime'].dt.hour
            df_usuario['month'] = df_usuario['datetime'].dt.month_name()
            df_usuario = df_usuario[['Consumo', 'day_of_week', 'Hour', 'month', 'Código universal de punto de suministro']]

            # NORMALIZACIÓN
            df_usuario[['day_of_week', 'month']] = enc_ord.transform(df_usuario[['day_of_week', 'month']])
            df_usuario = df_usuario.drop('Código universal de punto de suministro', axis=1)

            # PREDICCIÓN
            prediccion = gru_model.predict(df_usuario.to_numpy().reshape(1, 168, 4)).flatten()
            valores_usados = df_usuario["Consumo"]
            indices_tiempo = np.arange(len(valores_usados) + len(prediccion))
            plt.plot(indices_tiempo[:len(valores_usados)], valores_usados, label='Valores usados',)
            plt.plot(indices_tiempo[len(valores_usados):], prediccion, label='Predicción')
            plt.xlabel('Horas')
            plt.ylabel('Consumo')
            plt.legend()
            plt.title('Predicción del consumo')
            #plt.show()
            #print(prediccion)
            st.pyplot()
            st.markdown(prediccion)

        st.set_option('deprecation.showPyplotGlobalUse', False)

        # INTRODUCIR NUEVOS DATOS (EL USUARIO NOS PASA 168 DATOS)
        # Por simplicidad suponemos se pasan los 168 últimos datos de la predicción
        df = pd.read_csv("PCB.csv")
        df_usuario = df[len(df)-(7*24):]

        mu, std, enc_ord = requeriments_price()
        gru_model = tf.keras.models.load_model('gru_model_precio.h5')
        price_prediction(mu, std, enc_ord, gru_model, df_usuario)

        # INTRODUCIR NUEVOS DATOS (EL USUARIO NOS PASA 168 DATOS)
        # Por simplicidad suponemos que el usuario es el número 10 y nos da los valores de la última semana registrada
        df = pd.read_csv("electrodatos.csv")
        df_usuario = df[df['Código universal de punto de suministro'] == 10]
        df_usuario = df_usuario[len(df_usuario)-(7*24):]

        enc_ord = requeriments_consumo()
        gru_model = tf.keras.models.load_model('gru_model_consumo.h5')
        consumo_prediction(enc_ord, gru_model, df_usuario)

if __name__ == "__main__":
    torch.set_default_device("cuda")

    #Create model
    model = AutoModelForCausalLM.from_pretrained(
        "MILVLG/imp-v1-3b",
        torch_dtype=torch.float16,
        device_map=0,
        trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained("MILVLG/imp-v1-3b", trust_remote_code=True)
    main()
