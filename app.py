import joblib
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

# Muat model dan metrik
model = joblib.load('model_regresi_kafein.joblib')
metrics_dict = joblib.load('model_metrics.joblib')

# Muat DataFrame untuk menghitung nilai rata-rata dari data asli
df = pd.read_csv('caffeine_intake_tracker.csv')
df_cleaned = df[['caffeine_mg', 'focus_level']].dropna()

# PENTING: Hitung nilai rata-rata dari data ASLI
avg_caffeine_normalized = df_cleaned['caffeine_mg'].mean()

# Ambil batas persentil untuk klasifikasi adaptif
low_thresh = df_cleaned['focus_level'].quantile(0.33) * 100
med_thresh = df_cleaned['focus_level'].quantile(0.66) * 100

# Fungsi konversi skala untuk tampilan web
def to_real_caffeine(value):
    return value * 400

def to_real_score(value):
    return value * 100

app = Flask(__name__)

@app.route('/')
def home():
    display_metrics = {
        'mae': to_real_score(metrics_dict['mae']),
        'mse': to_real_score(metrics_dict['mse']),
        'rmse': to_real_score(metrics_dict['rmse']),
        'r2': metrics_dict['r2'],
        'intercept': to_real_score(metrics_dict['intercept']),
        'coefficient': metrics_dict['coefficient']
    }
    return render_template('index.html', metrics=display_metrics, avg_caffeine=to_real_caffeine(avg_caffeine_normalized))

@app.route('/predict_form')
def predict_form():
    return render_template('predict.html', avg_caffeine=to_real_caffeine(avg_caffeine_normalized))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        caffeine_input = float(request.form['caffeine_mg'])
        
        warning = None
        if caffeine_input > 400:
            warning = "PERINGATAN: Konsumsi kafein di atas 400 mg per hari dapat berdampak negatif pada kesehatan."

        normalized_input = caffeine_input / 400
        prediction = model.predict([[normalized_input]])
        predicted_focus = to_real_score(prediction[0])

        # 🔹 Klasifikasi adaptif berdasarkan distribusi data asli
        if predicted_focus < low_thresh:
            focus_category = "Rendah"
            focus_text = "Prediksi tingkat konsentrasi Anda tergolong **rendah**. Anda mungkin akan kesulitan fokus dan mudah terdistraksi."
        elif predicted_focus < med_thresh:
            focus_category = "Sedang"
            focus_text = "Prediksi tingkat konsentrasi Anda tergolong **sedang**. Anda masih bisa fokus, tetapi sesekali bisa terdistraksi."
        else:
            focus_category = "Tinggi"
            focus_text = "Prediksi tingkat konsentrasi Anda tergolong **tinggi**. Anda sangat fokus dan siap belajar dengan optimal."

        return render_template(
            'result.html',
            caffeine_input=caffeine_input,
            predicted_focus=predicted_focus,
            focus_category=focus_category,
            focus_text=focus_text,
            warning=warning
        )
    except ValueError:
        return "Input tidak valid. Silakan masukkan angka."

if __name__ == '__main__':
    app.run(debug=True)
