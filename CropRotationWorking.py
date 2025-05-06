import pandas as pd
import tensorflow as tf
import joblib

# ✅ Load the Trained Model and Encoders
model = tf.keras.models.load_model("crop_rotation_model.h5")
encoder = joblib.load("feature_encoder.pkl")
y_encoder = joblib.load("label_encoder.pkl")

# 🚜 Take Farmer Input
previous_crop = input("Enter your last crop (Rice/Wheat/Potato/Tomato/Corn/Barley/Peas/Soybean/Sugarcane/Cotton/Groundnut/Millet/Maize): ").strip().capitalize()
soil_type = input("Enter your soil type (Clay/Loam/Sandy): ").strip().capitalize()
season = input("Enter the season (Kharif/Rabi): ").strip().capitalize()

# Validate input
input_data = pd.DataFrame([[previous_crop, soil_type, season]], columns=["previous_crop", "soil_type", "season"])
input_encoded = encoder.transform(input_data).toarray()

# 🔮 AI Prediction
predicted_probs = model.predict(input_encoded)
predicted_crop = y_encoder.inverse_transform(predicted_probs).flatten()[0]

# 📌 **Dynamic Crop Rotation Reasoning**
reasoning = ""

# 1️⃣ **Nutrient Replenishment Logic**
if previous_crop in ["Wheat", "Corn", "Rice", "Maize", "Sugarcane"]:
    reasoning += "The previous crop depleted nitrogen, so the suggested crop helps restore soil fertility. "
elif previous_crop in ["Peas", "Lentils", "Soybean", "Groundnut"]:
    reasoning += "Since the previous crop fixed nitrogen, the suggested crop can utilize the enriched soil. "

# 2️⃣ **Soil Adaptation Logic**
if soil_type == "Clay":
    reasoning += "Clay soil retains moisture well, so the suggested crop is chosen for its ability to grow in such conditions. "
elif soil_type == "Sandy":
    reasoning += "Sandy soil drains quickly, so the suggested crop is drought-resistant and suited for dry conditions. "
elif soil_type == "Loam":
    reasoning += "Loam soil is balanced, making it suitable for a variety of crops, and the suggested crop makes the best use of this soil type. "

# 3️⃣ **Season Suitability Logic**
if season == "Kharif":
    reasoning += "The suggested crop thrives in the monsoon season, benefiting from ample water availability. "
elif season == "Rabi":
    reasoning += "The suggested crop is ideal for cooler temperatures and lower water availability. "

# 4️⃣ **Pest & Disease Management**
if (previous_crop, predicted_crop) in [
    ("Rice", "Wheat"), ("Wheat", "Lentils"), ("Peas", "Corn"), 
    ("Soybean", "Barley"), ("Maize", "Mustard"), ("Sugarcane", "Oats")
]:
    reasoning += "This rotation disrupts the cycle of pests and diseases, reducing the risk of infestations. "

# ✅ Default Reasoning If None of the Above Conditions Match
if not reasoning:
    reasoning = (
        "The suggested crop helps maintain soil health by improving organic matter content and reducing soil fatigue. "
        "It also supports better water and nutrient retention, ensuring sustainable productivity for the next season."
    )

print(f"\n🌾 **AI Recommended Next Crop:** {predicted_crop}")
print(f"📌 **Reason:** {reasoning}")
