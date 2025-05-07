from flask import Flask, render_template, request
import joblib
import hdbscan
import numpy as np

app = Flask(__name__)

model = joblib.load('hdbscan_model (1).joblib')
scaler = joblib.load('scaler (1).joblib')
feature_names = joblib.load('feature_names (1).joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            input_data = [float(request.form[name]) for name in feature_names]
        except KeyError as e:
            return f"Missing input for feature: {e.args[0]}", 400
        
        scaled_input = scaler.transform([input_data])
        
        cluster_label, strength = hdbscan.approximate_predict(model, scaled_input)
        
        return render_template('result.html', 
                             cluster=cluster_label[0],
                             strength=strength[0])
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
