from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import re

app = Flask(__name__)
CORS(app)

# 🔹 Load trained model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# 🔹 Scam keywords (extra safety layer)
scam_keywords = [
    "win", "lottery", "urgent", "click here", "free",
    "money", "prize", "otp", "account blocked",
    "verify now", "limited offer"
]

# 🔹 Home route (loads UI)
@app.route("/")
def home():
    return render_template("index.html")

# 🔹 Analyze route
@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    message = data.get("message", "")

    # Convert message for ML model
    msg_vec = vectorizer.transform([message])
    prediction = model.predict(msg_vec)[0]

    # Confidence score
    prob = model.predict_proba(msg_vec)[0]
    scam_prob = prob[list(model.classes_).index("scam")] * 100

    # Keyword flags
    flags = []
    text_lower = message.lower()

    for word in scam_keywords:
        if word in text_lower:
            flags.append(word)

    # Highlight risky words
    highlighted = message
    for word in flags:
        highlighted = re.sub(f"({word})", r"**\1**", highlighted, flags=re.IGNORECASE)

    # Explanation logic
    if prediction == "scam":
        explanation = "⚠️ This message shows common scam patterns like urgency, rewards, or suspicious links."
    else:
        explanation = "✅ This message appears safe, but always stay cautious."

    return jsonify({
        "prediction": prediction,
        "score": round(scam_prob, 2),
        "flags": flags,
        "highlighted": highlighted,
        "explanation": explanation
    })

# 🔹 Run app
if __name__ == "__main__":
    app.run(debug=True)