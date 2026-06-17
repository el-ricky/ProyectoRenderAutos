from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

modelo = joblib.load("modelo_autos.pkl")
columnas = joblib.load("columnas.pkl")


@app.route("/")
def inicio():
    return jsonify({
        "mensaje": "API funcionando correctamente",
        "modelo": "Random Forest"
    })


@app.route("/predecir", methods=["POST"])
def predecir():

    datos = request.get_json()

    df = pd.DataFrame([datos])

    for col in columnas:
        if col not in df.columns:
            df[col] = 0

    df = df[columnas]

    prediccion = modelo.predict(df)

    return jsonify({
        "precio_estimado": float(prediccion[0])
    })


if __name__ == "__main__":
    app.run(debug=True)