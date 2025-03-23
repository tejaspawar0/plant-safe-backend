from flask import Flask, render_template,request,jsonify
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
import os

# .\myenv\Scripts\activate
class_names = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

recommendations_text = {
    "apple_apple_scab": "🍏 Apple - Apple Scab: Remove bad leaves, prune branches to improve airflow, and apply captan fungicide.",
    "apple_black_rot": "🍏 Apple - Black Rot: Cut and destroy infected branches, remove mummified fruit, and apply copper fungicide.",
    "apple_cedar_apple_rust": "🍏 Apple - Cedar Apple Rust: Remove nearby cedar trees, prune infected branches, and spray myclobutanil fungicide.",
    "apple_healthy": "🍏 Apple - Healthy: Water regularly, prune dead branches, and monitor for any disease signs.",
    
    "blueberry_healthy": "🫐 Blueberry - Healthy: Maintain soil pH between 4.5-5.5, prune yearly, and ensure proper drainage.",

    "cherry_powdery_mildew": "🍒 Cherry - Powdery Mildew: Remove infected leaves, apply sulfur-based fungicide, and improve air circulation through pruning.",
    "cherry_healthy": "🍒 Cherry - Healthy: Ensure proper airflow, water at the base, and prune branches for better sunlight exposure.",

    "corn_cercospora_leaf_spot": "🌽 Corn - Cercospora Leaf Spot: Rotate crops yearly, remove infected leaves, and apply fungicide.",
    "corn_common_rust": "🌽 Corn - Common Rust: Plant resistant hybrids, avoid excessive nitrogen, and apply fungicides early if needed.",
    "corn_northern_leaf_blight": "🌽 Corn - Northern Leaf Blight: Remove infected leaves, improve field drainage, and apply fungicide.",
    "corn_healthy": "🌽 Corn - Healthy: Water deeply but avoid waterlogging, protect from pests, and rotate crops annually.",

    "grape_black_rot": "🍇 Grape - Black Rot: Remove diseased leaves and fruit mummies, spray mancozeb fungicide, and ensure good airflow.",
    "grape_esca_black_measles": "🍇 Grape - Esca (Black Measles): Prune infected wood, avoid overwatering, and keep vines well-ventilated.",
    "grape_leaf_blight": "🍇 Grape - Leaf Blight: Trim bad leaves, apply fungicide, and improve vineyard airflow.",
    "grape_healthy": "🍇 Grape - Healthy: Keep vines dry, prune regularly, and avoid excessive moisture.",

    "orange_huanglongbing": "🍊 Orange - Citrus Greening: Remove infected trees, control psyllid insects, and only plant disease-free citrus trees.",

    "peach_bacterial_spot": "🍑 Peach - Bacterial Spot: Remove infected leaves and fruits, apply copper fungicide, and plant resistant varieties.",
    "peach_healthy": "🍑 Peach - Healthy: Water properly, remove diseased leaves, and monitor for early signs of disease.",

    "pepper_bacterial_spot": "🫑 Bell Pepper - Bacterial Spot: Remove infected leaves, apply copper spray, and space plants properly to improve airflow.",
    "pepper_healthy": "🫑 Bell Pepper - Healthy: Maintain proper spacing, water adequately, and monitor for signs of disease.",

    "potato_early_blight": "🥔 Potato - Early Blight: Rotate crops, remove infected leaves, and spray chlorothalonil fungicide.",
    "potato_late_blight": "🥔 Potato - Late Blight: Destroy infected plants, apply metalaxyl-based fungicides, and avoid excess moisture.",
    "potato_healthy": "🥔 Potato - Healthy: Water at the base, avoid overwatering, and check leaves for any early disease symptoms.",

    "raspberry_healthy": "🍇 Raspberry - Healthy: Prune old canes, maintain good airflow, and remove diseased leaves.",

    "soybean_healthy": "🌱 Soybean - Healthy: Rotate crops yearly, ensure proper soil conditions, and check for pests.",

    "squash_powdery_mildew": "🎃 Squash - Powdery Mildew: Remove affected leaves, apply sulfur-based spray, and improve ventilation.",

    "strawberry_leaf_scorch": "🍓 Strawberry - Leaf Scorch: Trim bad leaves, apply fungicide, and water plants properly without wetting leaves.",
    "strawberry_healthy": "🍓 Strawberry - Healthy: Space plants well, mulch soil, and maintain proper watering habits.",

    "tomato_bacterial_spot": "🍅 Tomato - Bacterial Spot: Avoid handling wet plants, remove infected leaves, and apply copper spray.",
    "tomato_early_blight": "🍅 Tomato - Early Blight: Remove infected leaves, apply fungicide, and space plants properly.",
    "tomato_late_blight": "🍅 Tomato - Late Blight: Destroy infected plants, apply fungicide, and keep leaves dry.",
    "tomato_leaf_mold": "🍅 Tomato - Leaf Mold: Improve air circulation, remove affected leaves, and apply fungicide.",
    "tomato_septoria_leaf_spot": "🍅 Tomato - Septoria Leaf Spot: Remove lower leaves, apply fungicide, and keep leaves dry.",
    "tomato_spider_mites": "🍅 Tomato - Spider Mites: Use neem oil or insecticidal soap, increase humidity, and wash leaves with water.",
    "tomato_target_spot": "🍅 Tomato - Target Spot: Remove infected leaves, apply fungicide, and avoid overhead watering.",
    "tomato_yellow_leaf_curl_virus": "🍅 Tomato - Yellow Leaf Curl Virus: Control whiteflies, remove infected plants, and use resistant tomato varieties.",
    "tomato_tomato_mosaic_virus": "🍅 Tomato - Tomato Mosaic Virus: Avoid handling sick plants, remove infected ones, and disinfect tools.",
    "tomato_healthy": "🍅 Tomato - Healthy: Water at the base, use stakes for support, and inspect for early disease symptoms.",
    "corn_common_rust": "🌽 Corn - Common Rust: Plant resistant hybrids, avoid excessive nitrogen, and apply fungicides early if needed."
}

# Function to preprocess the image
def load_and_preprocess_image(img_path):
    # Load the image
    img = image.load_img(img_path, target_size=(128,128,3))  # Adjust target_size as per your model's input shape
    # Convert the image to an array
    img_array = image.img_to_array(img)
    # Expand dimensions to match the model's input shape
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the image array
  # Normalize to [0, 1] range if your model expects it
    return img_array

# Function to predict the disease
def predict_disease(img_path):
    # Preprocess the image
    img_array = load_and_preprocess_image(img_path)
    # Make predictions
    predictions = model.predict(img_array)
    # Get the class index with the highest probability
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "upload"
model = keras.models.load_model('CNN_model_30.h5')

@app.route("/prediction", methods=["POST"])
def prediction():
    if "img" not in request.files:
        return jsonify({"error": "No file received"}), 400

    img = request.files["img"]
    filename = secure_filename(img.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], "image." + filename.split(".")[-1])
    img.save(filepath)  # Save the uploaded image

    # Get prediction
    predicted_class = class_names[predict_disease(filepath)]
    
    # Format disease name for dictionary lookup
    formatted_disease = (
        predicted_class.lower()
        .replace(" ", "_")
        .replace("___", "_")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
    )

    print(f"Formatted Disease Key: {formatted_disease}")  # Debugging output

    # Get recommendation (without steps)
    recommendation_text = recommendations_text.get(formatted_disease, "No specific recommendation found.")

    response_data = {
        "disease": predicted_class,
        "recommendation": recommendation_text
    }

    print(f"Response Data: {response_data}")  # Debugging output

    return jsonify(response_data)


# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)