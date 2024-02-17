import base64
import io

import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

# Cargar los datos
df = pd.read_csv("electrodatos.csv")
df['Fecha'] = df['Fecha'].apply(lambda x: pd.Timestamp(x))
df['datetime'] = pd.to_datetime(df['datetime'])
df['day_of_week'] = df['datetime'].dt.day_name()

df['Hour'] = df['datetime'].dt.hour

df['month'] = df['datetime'].dt.month_name()

def asignar_categoria_tiempo(dt):
    hora = dt.hour
    if 6 <= hora < 14:
        return 'Morning'
    elif 14 <= hora < 22:
        return 'Afternoon'
    else:
        return 'Night'
    
df['Horario'] = df['datetime'].apply(asignar_categoria_tiempo)
dias_ordenados = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Colores personalizados
color_map = {
    'Morning': 'lightblue',
    'Afternoon': 'orange',
    'Night': 'darkblue'
}

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
    html.H2('Interactive color selection with simple Dash example'),
    
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
    dcc.Store(id='store'),
    html.Div([
            html.Button('¿Como interpreto mi factura?', id='submit-dos')
        ]),
    html.Div(id='output-data-upload'),

    # Gráfica 1
    dcc.Graph(
        id='mean-consumption-by-hour',
        figure=px.bar(df.groupby(['day_of_week', 'Hour'])['Consumo'].mean().unstack().reindex(dias_ordenados),
                      barmode='group', 
                      title='Consumo semanal', 
                      labels={'day_of_week': 'Day of the week', 'Consumo': 'Mean consumption (kWh)'},
                      category_orders={'day_of_week': dias_ordenados})
    ),

    # Gráfica 2
    dcc.Graph(
        id='mean-consumption-by-time-of-day',
        figure=px.bar(df.groupby(['day_of_week', 'Horario'])['Consumo'].mean().unstack().reindex(dias_ordenados), 
                      barmode='stack',
                      title='Consumo por días de la semana',
                      labels={'day_of_week': 'Day of the week', 'Consumo': 'Mean consumption (kWh)', 'Horario': 'Time of day'},
                      category_orders={'day_of_week': dias_ordenados},
                      color_discrete_map=color_map)
    )
])

@app.callback(Output('store', 'data'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(contents, list_of_names, list_of_dates):
    if contents is None:
        raise PreventUpdate

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)

    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    return df.to_json(date_format='iso', orient='split')


@app.callback(
    Output('output-data-upload', 'children'),
    Input('store', 'data')
)
def output_from_store(stored_data):
    df = pd.read_json(stored_data, orient='split')

    return html.Div([
        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns]
        ),
        html.Hr(),
    ])

# Dash code
if __name__ == '__main__':
  app.run_server(debug=True, threaded=True)
