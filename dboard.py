import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import joblib
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('car-dataset.csv')

# Initialize the Dash app with Bootstrap
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
            dbc.Col(dcc.Input(id='input-kil', type='number', value=20000), width=9)
        ]),
        dbc.Row([
            dbc.Col(html.Label('Mileage (kmpl):'), width=3),
            dbc.Col(dcc.Input(id='input-mil', type='number', value=20), width=9)
        ]),
        dbc.Row([
            dbc.Col(html.Label('Engine (CC):'), width=3),
            dbc.Col(dcc.Input(id='input-eng', type='number', value=1000), width=9)
        ]),
        dbc.Row([
            dbc.Col(html.Label('Power (HP):'), width=3),
            dbc.Col(dcc.Input(id='input-pow', type='number', value=100), width=9)
        ]),
        dbc.Row([
            dbc.Col(html.Label('Year:'), width=3),
            dbc.Col(dcc.Input(id='input-year', type='number', value=2015), width=9)
        ]),
        dbc.Row([
            dbc.Col(html.Label('Seats:'), width=3),
            dbc.Col(dcc.Dropdown(id='input-seats', 
                                  options=[{'label': i, 'value': i} for i in np.arange(1, 11)],
                                  value=5), width=9)
        ]),

        dbc.Row(
            dbc.Button('Predict', id='predict-button', color='primary', className="mt-3"),
            justify="center"
        ),
        html.H4(id='prediction-result', children='Predicted Price: ', className="mt-4 text-center")
    ],
    fluid=True
)

@app.callback(
    Output('prediction-result', 'children'),
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
        df = pd.concat([cat, input_num], axis=1)

        input = df.iloc[0, 0:]
        input = pd.DataFrame([input.values], columns=xgb.get_booster().feature_names)  
        
        predicted_price = xgb.predict(input)
        predicted_price = stand_scaler.inverse_transform(predicted_price.reshape(-1, 1))
        
        return f"Predicted Price: {predicted_price[0, 0]:.2f} Lakhs"

if __name__ == '__main__':
    app.run_server(debug=True)
