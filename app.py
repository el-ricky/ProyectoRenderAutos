from flask import Flask, jsonify
import joblib

app = Flask(__name__)

modelo = joblib.load("modelo_autos.pkl")
columnas = joblib.load("columnas.pkl")

@app.route("/")
def inicio():
    return jsonify({
        "mensaje": "API funcionando correctamente",
        "modelo": "Random Forest"
    })

if __name__ == "__main__":
    app.run(debug=True)