"""
app.py

Local UI for Heart Disease Prediction (Flask)

Run:
    python src/api/app.py
Then open:
    http://127.0.0.1:5000
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from flask import Flask, request, render_template_string
from src.models.inference import HeartDiseasePredictor

app = Flask(__name__)

predictor = HeartDiseasePredictor()

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Heart Disease Prediction</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f4f4f4;
            color: #333;
        }

        header {
            background: #b71c1c;
            color: white;
            padding: 14px 24px;
        }

        header h1 { font-size: 19px; font-weight: 600; }
        header p { font-size: 12px; opacity: 0.8; margin-top: 3px; }

        html { overflow: hidden; height: 100%; }
        body { height: 100%; }

        .wrap {
            max-width: 720px;
            margin: 14px auto;
            padding: 0 16px;
        }

        .card {
            background: white;
            border-radius: 8px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
            padding: 14px 20px;
            margin-bottom: 10px;
        }

        .card h2 {
            font-size: 13px;
            font-weight: 600;
            color: #b71c1c;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 2px solid #ffcdd2;
        }

        .row {
            display: flex;
            gap: 16px;
            margin-bottom: 8px;
        }

        .row .field { flex: 1; }

        .field label {
            display: block;
            font-size: 12px;
            font-weight: 500;
            color: #555;
            margin-bottom: 4px;
        }

        .field input,
        .field select {
            width: 100%;
            padding: 7px 10px;
            font-size: 13px;
            font-family: inherit;
            border: 1px solid #ccc;
            border-radius: 4px;
            background: #fafafa;
        }

        .field input:focus,
        .field select:focus {
            outline: none;
            border-color: #b71c1c;
            background: white;
        }

        .field input::placeholder { color: #aaa; }

        .btn-wrap {
            text-align: center;
            padding-top: 4px;
        }

        .btn {
            background: #b71c1c;
            color: white;
            border: none;
            padding: 10px 38px;
            font-size: 14px;
            font-weight: 500;
            border-radius: 4px;
            cursor: pointer;
        }

        .btn:hover { background: #9a1515; }

        .result {
            padding: 12px;
            border-radius: 6px;
            text-align: center;
        }

        .result-disease {
            background: #ffebee;
            border-left: 4px solid #e53935;
        }

        .result-healthy {
            background: #e8f5e9;
            border-left: 4px solid #43a047;
        }

        .result h3 { font-size: 16px; margin-bottom: 4px; }
        .result-disease h3 { color: #c62828; }
        .result-healthy h3 { color: #2e7d32; }

        .result p {
            font-size: 12px;
            color: #666;
            margin: 0;
        }
    </style>
</head>
<body>
    <header>
        <h1>Heart Disease Prediction</h1>
        <p>Enter patient data to assess heart disease risk</p>
    </header>

    <div class="wrap">
        <form method="POST">
            <div class="card">
                <h2>Demographics</h2>
                <div class="row">
                    <div class="field">
                        <label>Age</label>
                        <input type="number" name="age" placeholder="55" required>
                    </div>
                    <div class="field">
                        <label>Sex</label>
                        <select name="sex" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">Female</option>
                            <option value="1">Male</option>
                        </select>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>Clinical Measurements</h2>
                <div class="row">
                    <div class="field">
                        <label>Resting BP (mm Hg)</label>
                        <input type="number" name="trestbps" placeholder="130" required>
                    </div>
                    <div class="field">
                        <label>Cholesterol (mg/dl)</label>
                        <input type="number" name="chol" placeholder="250" required>
                    </div>
                </div>
                <div class="row">
                    <div class="field">
                        <label>Max Heart Rate</label>
                        <input type="number" name="thalach" placeholder="150" required>
                    </div>
                    <div class="field">
                        <label>ST Depression (Oldpeak)</label>
                        <input type="number" step="0.1" name="oldpeak" placeholder="1.5" required>
                    </div>
                </div>
                <div class="row">
                    <div class="field">
                        <label>Fasting Blood Sugar > 120</label>
                        <select name="fbs" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">No</option>
                            <option value="1">Yes</option>
                        </select>
                    </div>
                    <div class="field">
                        <label>Exercise Angina</label>
                        <select name="exang" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">No</option>
                            <option value="1">Yes</option>
                        </select>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>Cardiac Tests</h2>
                <div class="row">
                    <div class="field">
                        <label>Chest Pain Type</label>
                        <select name="cp" required>
                            <option value="" disabled selected>Select</option>
                            <option value="1">Typical Angina</option>
                            <option value="2">Atypical Angina</option>
                            <option value="3">Non-anginal Pain</option>
                            <option value="4">Asymptomatic</option>
                        </select>
                    </div>
                    <div class="field">
                        <label>Resting ECG</label>
                        <select name="restecg" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">Normal</option>
                            <option value="1">ST-T Abnormality</option>
                            <option value="2">LV Hypertrophy</option>
                        </select>
                    </div>
                </div>
                <div class="row">
                    <div class="field">
                        <label>Slope of ST Segment</label>
                        <select name="slope" required>
                            <option value="" disabled selected>Select</option>
                            <option value="1">Upsloping</option>
                            <option value="2">Flat</option>
                            <option value="3">Downsloping</option>
                        </select>
                    </div>
                    <div class="field">
                        <label>Major Vessels (0-3)</label>
                        <select name="ca" required>
                            <option value="" disabled selected>Select</option>
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="2">2</option>
                            <option value="3">3</option>
                        </select>
                    </div>
                </div>
                <div class="row">
                    <div class="field">
                        <label>Thalassemia</label>
                        <select name="thal" required>
                            <option value="" disabled selected>Select</option>
                            <option value="3">Normal</option>
                            <option value="6">Fixed Defect</option>
                            <option value="7">Reversible Defect</option>
                        </select>
                    </div>
                    <div class="field"></div>
                </div>
            </div>

            <div class="btn-wrap">
                <button type="submit" class="btn">Predict Risk</button>
            </div>
        </form>

        {% if result %}
        <div class="result {{ 'result-disease' if result.prediction == 1 else 'result-healthy' }}" style="margin-top:10px;">
            <h3>{{ result.prediction_label }}</h3>
            <p>Confidence: {{ "%.1f"|format(result.confidence * 100) }}% &bull; Risk Level: {{ result.risk_level }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        input_data = {
            'age': float(request.form["age"]),
            'sex': int(request.form["sex"]),
            'cp': int(request.form["cp"]),
            'trestbps': float(request.form["trestbps"]),
            'chol': float(request.form["chol"]),
            'fbs': int(request.form["fbs"]),
            'restecg': int(request.form["restecg"]),
            'thalach': float(request.form["thalach"]),
            'exang': int(request.form["exang"]),
            'oldpeak': float(request.form["oldpeak"]),
            'slope': int(request.form["slope"]),
            'ca': int(request.form["ca"]),
            'thal': int(request.form["thal"]),
        }
        result = predictor.predict(input_data)

    return render_template_string(HTML, result=result)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
