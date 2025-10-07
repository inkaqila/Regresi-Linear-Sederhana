import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import joblib

# Muat dataset
df = pd.read_csv('caffeine_intake_tracker.csv')

# Hapus kolom yang tidak relevan dan nilai yang tidak terpakai
df_cleaned = df[['caffeine_mg', 'focus_level']].dropna()

# Pisahkan variabel fitur (X) dan target (y)
X = df_cleaned[['caffeine_mg']]
y = df_cleaned['focus_level']

# Simpan nilai min dan max untuk scaling di app.py
scaling_params = {
    'min_caffeine': df_cleaned['caffeine_mg'].min(),
    'max_caffeine': df_cleaned['caffeine_mg'].max(),
    'min_focus': df_cleaned['focus_level'].min(),
    'max_focus': df_cleaned['focus_level'].max()
}
joblib.dump(scaling_params, 'scaling_params.joblib')

# Bagi data menjadi set pelatihan (training) dan pengujian (testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model Regresi Linear
model = LinearRegression()
model.fit(X_train, y_train)

# Buat scatterplot untuk visualisasi
plt.figure(figsize=(10, 6))
sns.scatterplot(x='caffeine_mg', y='focus_level', data=df_cleaned, label='Data Aktual')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Garis Regresi')
plt.title('Hubungan Konsumsi Kafein dan Tingkat Konsentrasi')
plt.xlabel('Konsumsi Kafein')
plt.ylabel('Tingkat Konsentrasi')
plt.grid(True)
plt.legend()
plt.savefig('static/scatterplot.png')

# Simpan model dan metrik
metrics_dict = {
    'mae': metrics.mean_absolute_error(y_test, model.predict(X_test)),
    'mse': metrics.mean_squared_error(y_test, model.predict(X_test)),
    'rmse': np.sqrt(metrics.mean_squared_error(y_test, model.predict(X_test))),
    'r2': metrics.r2_score(y_test, model.predict(X_test)),
    'intercept': model.intercept_,
    'coefficient': model.coef_[0]
}
joblib.dump(model, 'model_regresi_kafein.joblib')
joblib.dump(metrics_dict, 'model_metrics.joblib')

print("Semua file telah berhasil disimpan.")