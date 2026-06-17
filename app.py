from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Cargar modelo y columnas
modelo = joblib.load("modelo_autos.pkl")
columnas = joblib.load("columnas.pkl")


# Mostrar la página principal
@app.route("/")
def inicio():
    return render_template("index.html")


# Endpoint para predecir
@app.route("/predecir", methods=["POST"])
def predecir():

    datos = request.get_json()

    df = pd.DataFrame([datos])

    # Agregar columnas faltantes
    for col in columnas:
        if col not in df.columns:
            df[col] = 0

    # Ordenar columnas como fueron entrenadas
    df = df[columnas]

    # Predicción
    prediccion = modelo.predict(df)

    return jsonify({
        "precio_estimado": round(float(prediccion[0]), 2)
    })


if __name__ == "__main__":
    app.run(debug=True)