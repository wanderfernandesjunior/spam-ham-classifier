import numpy as np

# Bibliotecas do Flask
from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request

# Biblioteca para carregar o modelo
import joblib

def create_app(test_config=None):

    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            texto = request.form.get("texto")

            # ler o modelo
            clf_pipeline = joblib.load('modelo.pkl')
            resultado = "Spam" if clf_pipeline.predict([texto])[0] else "Ham"
            probabilidade  = np.max(clf_pipeline.predict_proba([texto]))

            return jsonify(resultado=resultado, probabilidade=round(probabilidade*100, 2)), 200
        
        except Exception as erro:
            return jsonify(erro=str(erro)), 500

    return app
