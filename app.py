from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import flask
import io

# Initialize Flask app and load the model
app = flask.Flask(__name__)
model = load_model("skin_cancer_detection_model.h5")


@app.route("/predict", methods=["POST"])
def predict():
    if flask.request.method == "POST":
        # Check if an image file is present in the request
        if 'file' not in flask.request.files:
            return "No file found", 400
        file = flask.request.files["file"]
        
        # Load and preprocess the image
        if file:
            image = Image.open(io.BytesIO(file.read()))
            image = image.resize((32, 32))  # Resize to match model's input
            image = img_to_array(image) / 255.0
            image = np.expand_dims(image, axis=0)  # Expand dimensions

            # Make a prediction
            preds = model.predict(image)
            class_idx = np.argmax(preds, axis=1)[0]
            class_labels = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
            label = class_labels[class_idx]

            # Return the result as JSON
            return flask.jsonify({"class": label, "confidence": float(preds[0][class_idx])})


if __name__ == "__main__":
    app.run(debug=True)
