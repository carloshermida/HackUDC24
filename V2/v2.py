import base64
import io

import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px

# App de Dash
app = dash.Dash(__name__)

# Layout de la aplicación
app.layout = html.Div(children=[
    dash.html.Header(
      children=[
        html.Div([
            html.Img(src='/assets/logo.png')
        ]),
        html.Div([
            html.H1('¿Cómo interpreto mi factura de la luz?')
        ])
      ]),
    html.H2('Introduce aqui tu factura de la luz:'),
    
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
    ),
    html.P(id='upload-feedback'),

    dcc.Store(id='store'),

    html.Div(id='texto-confirm', children=[

    ]),

    html.Div([
            html.Button('¿Como interpreto mi factura?', id='submit-dos'),
            html.Button('¿Cómo ahorro en mi factura de la luz?', id='submit-tres')
    ]),
    # División para contener las visualizaciones
    html.Div(id='visualizaciones', style={'display': 'none'}, children=[
        # Gráfica 1
        dcc.Graph(
            id='mean-consumption-by-hour'
        ),

        # Gráfica 2
        dcc.Graph(
            id='mean-consumption-by-time-of-day'
        )
    ]),

])

# Callback para actualizar el contenido del almacenamiento con los datos del archivo cargado
@app.callback([Output('store', 'data'),
              Output('texto-confirm', 'children')],
              Input('upload-data', 'contents'),
              State('upload-data', 'filename')) 
def update_upload_feedback(contents, filename):
    if contents is None:
        return ''  # No se ha cargado ningún archivo, no hay feedback

    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

        df['Fecha'] = df['Fecha'].apply(lambda x: pd.Timestamp(x))
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['day_of_week'] = df['datetime'].dt.day_name()

        df['Hour'] = df['datetime'].dt.hour
        df['month'] = df['datetime'].dt.month_name()
        df['Horario'] = df['datetime'].apply(asignar_categoria_tiempo)
        
        return df.to_json(date_format='iso', orient='split'), [html.P("El archivo se ha cargado correctamente!", style={'color': 'green'})]
    except Exception as e:
        return "", [html.P('Error al cargar el archivo: {}'.format(filename), style={'color': 'red'})]

# Callback para mostrar las visualizaciones cuando se presiona el botón correspondiente
@app.callback(
    Output('visualizaciones', 'style'),
    Input('submit-dos', 'n_clicks')
)
def show_hide_visualizations(n_clicks):
    if n_clicks is None or n_clicks == 0:
        return {'display': 'none'}  # Ocultar visualizaciones si el botón no ha sido presionado
    else:
        return {'display': 'block'}  # Mostrar visualizaciones si el botón ha sido presionado

# Callback para generar las gráficas
@app.callback(
    Output('mean-consumption-by-hour', 'figure'),
    Output('mean-consumption-by-time-of-day', 'figure'),
    Input('store', 'data')
)
def update_charts(stored_data):
    if not stored_data:
        raise PreventUpdate
    
    df = pd.read_json(stored_data, orient='split')
    dias_ordenados = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Colores personalizados
    color_map = {
        'Morning': 'lightblue',
        'Afternoon': 'orange',
        'Night': 'darkblue'
    }

    fig1 = px.bar(df.groupby(['day_of_week', 'Hour'])['Consumo'].mean().unstack().reindex(dias_ordenados),
                  barmode='group', 
                  title='Consumo semanal', 
                  labels={'day_of_week': 'Day of the week', 'Consumo': 'Mean consumption (kWh)'},
                  category_orders={'day_of_week': dias_ordenados})

    fig2 = px.bar(df.groupby(['day_of_week', 'Horario'])['Consumo'].mean().unstack().reindex(dias_ordenados), 
                  barmode='stack',
                  title='Consumo por días de la semana',
                  labels={'day_of_week': 'Day of the week', 'Consumo': 'Mean consumption (kWh)', 'Horario': 'Time of day'},
                  category_orders={'day_of_week': dias_ordenados},
                  color_discrete_map=color_map)

    return fig1, fig2

# Definir función para asignar categoría de tiempo
def asignar_categoria_tiempo(dt):
    hora = dt.hour
    if 6 <= hora < 14:
        return 'Morning'
    elif 14 <= hora < 22:
        return 'Afternoon'
    else:
        return 'Night'

# Dash code
if __name__ == '__main__':
  app.run_server(debug=True, threaded=True)
