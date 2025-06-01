from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import os
import joblib
from automl.preprocessing import preprocess_data

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and preprocessor once on startup
model = joblib.load("outputs/best_model.pkl")
preprocessor = joblib.load("outputs/preprocessor.pkl")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Load new data
            data = pd.read_csv(filepath)

            # Preprocess data (prediction mode)
            global preprocess_data
            # Manually inject preprocessor for this session
            preprocess_data.preprocessor = preprocessor

            X = preprocess_data(data, training=False)
            predictions = model.predict(X)

            # Add predictions to dataframe
            data['Prediction'] = predictions
            output_file = os.path.join(app.config['UPLOAD_FOLDER'], 'predictions.csv')
            data.to_csv(output_file, index=False)

            return render_template('result.html', tables=[data.head().to_html(classes='data')], filename='predictions.csv')

    return render_template('upload.html')

@app.route('/download/<filename>')
def download_file(filename):
    return redirect(url_for('static', filename=filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)
