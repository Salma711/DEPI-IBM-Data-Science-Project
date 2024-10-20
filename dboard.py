import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


df = pd.read_csv('car-dataset.csv')

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY])

app.layout = dbc.Container(
    [
        html.H1("Used Car Price Analysis and Prediction", className="text-center my-4"),

        html.H3("Predict Car Price"),
        dbc.Row([
            dbc.Col(html.Label('Location:'), width=3),
            dbc.Col(dcc.Dropdown(id='input-loc', 
                                  options=[{'label': i, 'value': i} for i in df['Location'].unique()],
                                  value='Delhi'), width=9)
        ]),
        dbc.Row([
            dbc.Col(html.Label('Fuel Type:'), width=3),
            dbc.Col(dcc.Dropdown(id='input-fuel', 
                                  options=[{'label': i, 'value': i} for i in df['Fuel_Type'].unique()],
                                  value='Petrol'), width=9)
        ]),
        dbc.Row([
            dbc.Col(html.Label('Transmission:'), width=3),
            dbc.Col(dcc.Dropdown(id='input-transmission', 
                                  options=[{'label': i, 'value': i} for i in df['Transmission'].unique()],
                                  value='Manual'), width=9)
        ]),
        dbc.Row([
            dbc.Col(html.Label('Brand:'), width=3),
            dbc.Col(dcc.Dropdown(id='input-brand', 
                                  options=[{'label': i, 'value': i} for i in df['Brand'].unique()],
                                  value='Maruti'), width=9)
        ]),
        dbc.Row([
            dbc.Col(html.Label('Model:'), width=3),
            dbc.Col(dcc.Dropdown(id='input-model', 
                                  options=[{'label': i, 'value': i} for i in df['Model'].unique()],
                                  value='Wagon R'), width=9)
        ]),
        dbc.Row([
            dbc.Col(html.Label('Owner Type:'), width=3),
            dbc.Col(dcc.Dropdown(id='input-owner', 
                                  options=[{'label': i, 'value': i} for i in df['Owner_Type'].unique()],
                                  value='First'), width=9)
        ]),
        dbc.Row([
            dbc.Col(html.Label('Kilometers Driven:'), width=3),
            dbc.Col(
                dcc.Slider(
                    id='input-kil',
                    min=0,
                    max=700000,
                    step=1000,   
                    value=50000, 
                    marks={i: str(i) for i in range(0, 700001, 50000)}, 
                ), 
                width=9
            )
        ]),
        dbc.Row(
            dbc.Col(html.Div(id='output-kil', className="text-center mt-2"), width=12)
        ),
        dbc.Row([
            dbc.Col(html.Label('(kmpl):'), width=3),
            dbc.Col(
                dcc.Slider(
                    id='input-mil',
                    min=0,
                    max=50,
                    step=1,   
                    value=20, 
                    marks={i: str(i) for i in range(0, 51, 10)},  
                ), 
                width=9
            )
        ]),
        dbc.Row(
            dbc.Col(html.Div(id='output-mil', className="text-center mt-2"), width=12)
        ),
        dbc.Row([
            dbc.Col(html.Label('Engine (CC):'), width=3),
            dbc.Col(
                dcc.Slider(
                    id='input-eng',
                    min=0,
                    max=6000,
                    step=100,   
                    value=1000, 
                    marks={i: str(i) for i in range(0, 6001, 500)},  
                ), 
                width=9
            )
        ]),
        dbc.Row(
            dbc.Col(html.Div(id='output-eng', className="text-center mt-2"), width=12)
        ),
        dbc.Row([
            dbc.Col(html.Label('Power (HP):'), width=3),
            dbc.Col(
                dcc.Slider(
                    id='input-pow',
                    min=0,
                    max=600,
                    step=10,   
                    value=100, 
                    marks={i: str(i) for i in range(0, 601, 50)},  
                ), 
                width=9
            )
        ]),
        dbc.Row(
            dbc.Col(html.Div(id='output-pow', className="text-center mt-2"), width=12)
        ),
        dbc.Row([
            dbc.Col(html.Label('Year:'), width=3),
            dbc.Col(
                dcc.Slider(
                    id='input-year',
                    min=2000,
                    max=2024,
                    step=1,   
                    value=2015, 
                    marks={i: str(i) for i in range(2000, 2025, 2)},  
                ), 
                width=9
            )
        ]),
        dbc.Row(
            dbc.Col(html.Div(id='output-year', className="text-center mt-2"), width=12)
        ),
        dbc.Row([
            dbc.Col(html.Label('Seats:'), width=3),
            dbc.Col(
                dcc.Slider(
                    id='input-seats',
                    min=0,
                    max=10,
                    step=1,   
                    value=5, 
                    marks={i: str(i) for i in range(0, 11, 1)},  
                ), 
                width=9
            )
        ]),
        dbc.Row(
            dbc.Col(html.Div(id='output-seats', className="text-center mt-2"), width=12)
        ),
        
        dbc.Row(
            dbc.Button('Predict', id='predict-button', color='primary', className="mt-3"),
            justify="center"
        ),
        html.H4(id='prediction-result', children='Predicted Price: ', className="mt-4 text-center"),
        dbc.Row(
            dbc.Col(dcc.Graph(id='price-comparison-graph', className="mt-4"), width=12)
        )
    ],
    fluid=True
)
@app.callback(
    Output('output-kil', 'children'),
    [Input('input-kil', 'value')]
)
def update_kilometers_display(kil):
    return f'Selected Kilometers: {kil}'

@app.callback(
    Output('output-mil', 'children'),
    [Input('input-mil', 'value')]
)
def update_mil_display(mil):
    return f'Selected Kmpl: {mil}'

@app.callback(
    Output('output-eng', 'children'),
    [Input('input-eng', 'value')]
)
def update_eng_display(eng):
    return f'Selected Engine: {eng}'

@app.callback(
    Output('output-pow', 'children'),
    [Input('input-pow', 'value')]
)
def update_pow_display(pow):
    return f'Selected Power: {pow}'

@app.callback(
    Output('output-year', 'children'),
    [Input('input-year', 'value')]
)
def update_year_display(year):
    return f'Selected Year: {year}'

@app.callback(
    Output('output-seats', 'children'),
    [Input('input-seats', 'value')]
)
def update_seats_display(seats):
    return f'Selected Seats: {seats}'

@app.callback(
    [Output('prediction-result', 'children'),
     Output('price-comparison-graph', 'figure')],
    [Input('predict-button', 'n_clicks')],
    [Input('input-loc', 'value'),
     Input('input-fuel', 'value'),
     Input('input-transmission', 'value'),
     Input('input-brand', 'value'),
     Input('input-model', 'value'),
     Input('input-owner', 'value'),
     Input('input-kil', 'value'),
     Input('input-mil', 'value'),
     Input('input-eng', 'value'),
     Input('input-pow', 'value'),
     Input('input-year', 'value'),
     Input('input-seats', 'value')]
)
def predict_price(n_clicks, loc, fuel, transmission, brand, model, owner, kil, mil, eng, pow, year, seats):
    if n_clicks:
        xgb = joblib.load('car_price_model.pkl')
        stand_scaler = joblib.load('stand_scaler.pkl')
        robust_scaler = joblib.load('robust_scaler.pkl')
        brand_encoder = joblib.load('brand_encoder.pkl')
        model_encoder = joblib.load('model_encoder.pkl')
        owner_encoder = joblib.load('owner_encoder.pkl')
        one_hot = joblib.load('one_hot.pkl')

        input_data = pd.DataFrame([[loc.lower(), fuel.lower(), transmission.lower(), brand.lower(), model.lower(), owner.lower(), kil, mil, eng, pow, year, seats]],
                                  columns=['Location', 'Fuel_Type', 'Transmission', 'Brand', 'Model', 'Owner_Type', 'Kilometers_Driven', 'kmpl', 'CC', 'Horse_Power','Year', 'Seats'])
        
        input_num = input_data[['Kilometers_Driven', 'kmpl', 'CC', 'Horse_Power','Year', 'Seats']]
        input_num = robust_scaler.transform(input_num)
        input_num = pd.DataFrame(input_num, columns=robust_scaler.get_feature_names_out())
        
        input_label = input_data[["Brand", "Model", "Owner_Type"]]
        input_label['Brand'] = brand_encoder.transform(input_label['Brand'])
        input_label['Model'] = model_encoder.transform(input_label['Model'])
        input_label['Owner_Type'] = owner_encoder.transform(input_label['Owner_Type'])

        input_one_hot = input_data[['Location', 'Fuel_Type', 'Transmission']]
        input_one_hot = one_hot.transform(input_one_hot).toarray()
        input_one_hot = pd.DataFrame(input_one_hot, columns=one_hot.get_feature_names_out())
        
        cat = pd.concat([input_one_hot, input_label], axis=1)
        df_in = pd.concat([cat, input_num], axis=1)

        input = df_in.iloc[0, 0:]
        input = pd.DataFrame([input.values], columns=xgb.get_booster().feature_names)  
        
        predicted_price = xgb.predict(input)
        predicted_price = stand_scaler.inverse_transform(predicted_price.reshape(-1, 1))[0,0]
        
        df_location = df[(df['Location'] == loc) & (df['Year'] == year)]

        
        fig = px.box(df_location, y='Price', title=f'Price Distribution for Cars in {loc} ({year})',
                     labels={'Price': 'Price (Lakhs)'})

    
        fig.add_trace(go.Scatter(x=[0], y=[predicted_price], mode='markers', 
                                 marker=dict(color='red', size=10), name='Predicted Price'))

        return f"Predicted Price: {predicted_price:.2f} Lakhs", fig
    return 'Predicted Price: ', {}
if __name__ == '__main__':
    app.run_server(debug=True)
