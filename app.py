from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# ✅ NEW (NGROK)
from pyngrok import ngrok

app = Flask(__name__)
app.secret_key = "secret123"

# ---------------- DATABASE ----------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

with app.app_context():
    db.create_all()

# ---------------- UPLOAD ----------------
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ---------------- MODEL ----------------
model = load_model("chili_leaf_model.h5")

class_names = [
    "Bacterial Spot",
    "Cercospora Leaf Spot",
    "Curl Virus",
    "Healthy Leaf",
    "Nutrition Deficiency",
    "White Spot"
]

# ---------------- MULTI LANGUAGE + QUANTITY ----------------
disease_info = {
    "White Spot": {
        "en": {"cause": "Fungal infection due to humidity.", "treatment": "Use fungicide.", "quantity": "Apply 2g per liter (~200L per acre)."},
        "hi": {"cause": "फंगल संक्रमण।", "treatment": "फफूंदनाशी का उपयोग करें।", "quantity": "200 लीटर पानी में 2 ग्राम दवा प्रति एकड़।"},
        "te": {"cause": "ఫంగస్ సంక్రమణ.", "treatment": "ఫంగిసైడ్ వాడాలి.", "quantity": "ఒక ఎకరానికి 200 లీటర్లలో 2 గ్రాములు కలపాలి."}
    },
    "Nutrition Deficiency": {
        "en": {"cause": "Lack of nutrients.", "treatment": "Use fertilizers.", "quantity": "Apply 25-50 kg fertilizer per acre."},
        "hi": {"cause": "पोषक तत्वों की कमी।", "treatment": "उर्वरक का उपयोग करें।", "quantity": "प्रति एकड़ 25-50 किलो उर्वरक डालें।"},
        "te": {"cause": "పోషక లోపం.", "treatment": "ఎరువులు వాడాలి.", "quantity": "ఒక ఎకరానికి 25-50 కిలోల ఎరువు వాడాలి."}
    },
    "Healthy Leaf": {
        "en": {"cause": "Healthy leaf.", "treatment": "No treatment needed.", "quantity": "No chemicals required."},
        "hi": {"cause": "स्वस्थ पत्ता।", "treatment": "कोई उपचार नहीं।", "quantity": "कोई दवा आवश्यक नहीं।"},
        "te": {"cause": "ఆరోగ్యకరమైన ఆకు.", "treatment": "చికిత్స అవసరం లేదు.", "quantity": "ఎటువంటి మందు అవసరం లేదు."}
    },
    "Curl Virus": {
        "en": {"cause": "Virus via whiteflies.", "treatment": "Use neem oil.", "quantity": "3-5 ml neem oil per liter (~150L per acre)."},
        "hi": {"cause": "वायरस संक्रमण।", "treatment": "नीम तेल का उपयोग करें।", "quantity": "3-5 ml प्रति लीटर (~150 लीटर/एकड़)।"},
        "te": {"cause": "వైరస్.", "treatment": "నీమ్ ఆయిల్ వాడాలి.", "quantity": "ఒక లీటర్‌కు 3-5 ml (~150 లీటర్లు/ఎకరం)."}
    },
    "Cercospora Leaf Spot": {
        "en": {"cause": "Fungal disease.", "treatment": "Use fungicide.", "quantity": "2g per liter (~200L per acre)."},
        "hi": {"cause": "फंगल रोग।", "treatment": "फफूंदनाशी का उपयोग करें।", "quantity": "200 लीटर पानी में 2 ग्राम प्रति एकड़।"},
        "te": {"cause": "ఫంగస్ వ్యాధి.", "treatment": "ఫంగిసైడ్ వాడాలి.", "quantity": "ఒక ఎకరానికి 200 లీటర్లలో 2 గ్రాములు కలపాలి."}
    },
    "Bacterial Spot": {
        "en": {"cause": "Bacterial infection.", "treatment": "Use bactericide.", "quantity": "2-3g per liter (~200L per acre)."},
        "hi": {"cause": "बैक्टीरिया संक्रमण।", "treatment": "दवा का उपयोग करें।", "quantity": "2-3 ग्राम प्रति लीटर (~200 लीटर/एकड़)।"},
        "te": {"cause": "బ్యాక్టీరియా.", "treatment": "మందు వాడాలి.", "quantity": "ఒక లీటర్‌కు 2-3 గ్రాములు (~200 లీటర్లు/ఎకరం)."}
    }
}

# ---------------- PREDICTION ----------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)

    return predicted_class, confidence

# ---------------- LOGIN ----------------
@app.route("/", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        user = User.query.filter_by(username=request.form["username"]).first()
        if user and check_password_hash(user.password, request.form["password"]):
            session["user"] = user.username
            return redirect(url_for("dashboard"))
        else:
            error = "Invalid Credentials"
    return render_template("login.html", error=error)

# ---------------- SIGNUP ----------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        if User.query.filter_by(username=request.form["username"]).first():
            return "User already exists"
        user = User(
            username=request.form["username"],
            password=generate_password_hash(request.form["password"])
        )
        db.session.add(user)
        db.session.commit()
        return redirect(url_for("login"))
    return render_template("signup.html")

# ---------------- FORGOT ----------------
@app.route("/forgot")
def forgot():
    return render_template("forgot.html")

# ---------------- DASHBOARD ----------------
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():

    if "user" not in session:
        return redirect(url_for("login"))

    prediction = None
    confidence = None
    image_path = None
    info = None

    if request.method == "POST":
        file = request.files["file"]

        if file and file.filename != "":
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            prediction, confidence = predict_image(filepath)
            image_path = filepath

            info = disease_info.get(prediction)

    return render_template("dashboard.html",
                           prediction=prediction,
                           confidence=confidence,
                           image_path=image_path,
                           info=info)

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# ---------------- RUN ----------------
if __name__ == "__main__":
    # Start ngrok tunnel
    public_url = ngrok.connect(5000)
    print(" Public URL:", public_url)

    # Run Flask
    app.run(host="0.0.0.0", port=5000, debug=True)