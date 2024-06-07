# Import cần thiết
from flask import Flask, render_template, request, redirect, url_for, send_file
import pandas as pd
import joblib
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import io
import os

app = Flask(__name__)

# Load models
models = {}
model_files = {
    'Random Forest': 'random_forest.pkl',
    'Naive Bayes': 'nb_model.pkl',
    'K-Nearest Neighbors': 'knn_model.pkl',
    'Decision Tree': 'dt_model.pkl'
}

for model_name, model_file in model_files.items():
    if os.path.exists(model_file):
        models[model_name] = joblib.load(model_file)
    else:
        print(f"Warning: {model_file} does not exist. {model_name} model will not be available.")

# Define the feature columns
feature_columns = [
    'basket_icon_click', 'basket_add_list', 'basket_add_detail', 'sort_by', 
    'image_picker', 'account_page_click', 'promo_banner_click', 'detail_wishlist_add', 
    'list_size_dropdown', 'closed_minibasket_click', 'checked_delivery_detail', 
    'checked_returns_detail', 'sign_in', 'saw_checkout', 'saw_sizecharts', 
    'saw_delivery', 'saw_account_upgrade', 'saw_homepage', 'device_computer', 
    'device_tablet', 'returning_user', 'loc_uk'
]

# Global variable to store uploaded dataframe
uploaded_df = None

# Create Dash app
dash_app = Dash(__name__, server=app, url_base_pathname='/dashboard/')

# Dash layout
dash_app.layout = html.Div([
    html.H1('Customer Data Executive Dashboard'),
    html.Div([
        dcc.Tabs(id="tabs", value='tab-overview', children=[
            dcc.Tab(label='Overview', value='tab-overview'),
            dcc.Tab(label='Detailed View', value='tab-detailed')
        ]),
        html.Div(id='tabs-content')
    ])
])

@dash_app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    global uploaded_df
    if uploaded_df is not None:
        if tab == 'tab-overview':
            total_customers = len(uploaded_df)
            potential_customers = len(uploaded_df[uploaded_df['Score'] > 0.8])
            potential_rate = potential_customers / total_customers

            fig_pie = go.Figure(data=[go.Pie(labels=['Potential', 'Not Potential'], 
                                             values=[potential_customers, total_customers - potential_customers])])
            fig_pie.update_layout(title='Customer Potential Distribution')

            fig_ordered = go.Figure(data=[go.Pie(labels=uploaded_df['ordered'].value_counts().index.tolist(), 
                                                 values=uploaded_df['ordered'].value_counts().values.tolist())])
            fig_ordered.update_layout(title='Ordered Distribution')

            return html.Div([
                html.Div([
                    html.H3('Total Customers'),
                    html.P(f'{total_customers}')
                ], className='card'),
                html.Div([
                    html.H3('Potential Customers'),
                    html.P(f'{potential_customers}')
                ], className='card'),
                html.Div([
                    html.H3('Potential Rate'),
                    html.P(f'{potential_rate:.2%}')
                ], className='card'),
                dcc.Graph(figure=fig_pie),
                dcc.Graph(figure=fig_ordered)
            ])
        elif tab == 'tab-detailed':
            graphs = []
            for column in feature_columns + ['ordered']:
                if column in uploaded_df.columns:
                    fig = go.Figure(data=[go.Pie(labels=uploaded_df[column].value_counts().index.tolist(), 
                                                 values=uploaded_df[column].value_counts().values.tolist())])
                    fig.update_layout(title=f'Distribution of {column}')
                    graphs.append(dcc.Graph(figure=fig))
            return html.Div(graphs)
    return html.Div()

@app.route('/')
def index():
    return render_template('index.html', models=models.keys(), feature_columns=feature_columns)

@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form['model']
    model = models[model_name]
    input_data = {feature: int(request.form[feature]) for feature in feature_columns}
    input_df = pd.DataFrame([input_data])
    score = model.predict_proba(input_df)[:, 1][0]
    result = "Potential Customer" if score > 0.8 else "Not a Potential Customer"
    return render_template('result.html', model_name=model_name, score=score, result=result)

@app.route('/upload', methods=['POST'])
def upload_file():
    global uploaded_df
    model_name = request.form['model']
    model = models[model_name]
    file = request.files['file']
    if not file:
        return redirect(url_for('index'))

    # Read the uploaded file into a dataframe
    file_stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
    uploaded_df = pd.read_csv(file_stream)
    
    df_features = uploaded_df[feature_columns]
    uploaded_df['Score'] = model.predict_proba(df_features)[:, 1]
    potential_customers = uploaded_df[uploaded_df['Score'] > 0.8]
    output = io.BytesIO()
    potential_customers.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='potential_customers.csv')

if __name__ == '__main__':
    app.run(debug=True)
