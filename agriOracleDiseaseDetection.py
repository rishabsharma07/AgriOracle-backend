import os
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
import matplotlib.pyplot as plt # type: ignore

# Load model
model = load_model(r"C:\Users\Param\Desktop\Krishi\Model\all_3_classification_mobilenetv2.h5")
img_size = (224, 224)

# Class labels
class_labels = [
    "Cotton Bacterial Blight", "Cotton Boll Rot", "Cotton Healthy", "Cotton Mildew",
    "Potato Common Scab", "Potato Dry Rot", "Potato Healthy", "Potato Scab",
    "Rice Leaf Blight", "Rice Healthy", "Rice Leaf Scald", "Rice Sheath Blight",
    "Wheat Black Rust", "Wheat Head Blight", "Wheat Healthy", "Wheat Mildew", "Wheat Smut"
]

# Reasons for diseases
disease_reasons = {
    "Cotton Bacterial Blight": "Caused by the bacterium *Xanthomonas citri*, spread through infected seeds, rain splash, and high humidity.",
    "Cotton Boll Rot": "Develops due to fungal or bacterial infections, favored by wet conditions during boll formation.",
    "Cotton Healthy": "Plant is healthy; proper field management and seed treatment maintained.",
    "Cotton Mildew": "Caused by fungi like *Leveillula taurica*, thriving in humid conditions with poor air circulation.",
    "Potato Common Scab": "Caused by *Streptomyces* bacteria in alkaline soils with low moisture.",
    "Potato Dry Rot": "Fungal disease caused by *Fusarium* species, infecting tubers through wounds during harvest or storage.",
    "Potato Healthy": "Plant is healthy; good agricultural practices followed.",
    "Potato Scab": "Develops in dry soils with a high pH; caused by *Streptomyces scabies* bacteria.",
    "Rice Leaf Blight": "Caused by *Xanthomonas oryzae* bacteria, often triggered by storms and strong winds damaging leaves.",
    "Rice Healthy": "Plant is healthy; proper irrigation and seed selection maintained.",
    "Rice Leaf Scald": "Fungal disease favored by poor soil nutrition and fluctuating weather conditions.",
    "Rice Sheath Blight": "Caused by *Rhizoctonia solani* fungus, thriving in dense plantings and high humidity.",
    "Wheat Black Rust": "Caused by *Puccinia graminis* fungus, spreads through wind-borne spores during warm, humid conditions.",
    "Wheat Head Blight": "Fungal infection caused by *Fusarium* species, promoted by wet, warm weather during flowering.",
    "Wheat Healthy": "Crop is healthy; timely fungicide use and crop rotation practices applied.",
    "Wheat Mildew": "Caused by *Blumeria graminis*, favored by cool, damp weather and overcrowding.",
    "Wheat Smut": "Fungal disease caused by *Tilletia* species, spreading through infected seeds."
}

# Cure or Treatment Recommendations (detailed)
cure_recommendations = {
    "Cotton Bacterial Blight": """• Use certified disease-free seeds.
• Practice 2–3 year crop rotation with non-host crops.
• Remove and burn infected plants.
• Apply copper-based bactericides at early stages.
• Avoid overhead irrigation to minimize leaf wetness.""",
    
    "Cotton Boll Rot": """• Improve plant spacing for better air circulation.
• Apply appropriate fungicides during flowering.
• Remove damaged or rotting bolls promptly.
• Avoid excessive nitrogen fertilizers.""",
    
    "Cotton Healthy": "• Continue proper irrigation, pest monitoring, and field hygiene.",
    
    "Cotton Mildew": """• Apply sulfur-based or systemic fungicides as preventive sprays.
• Ensure good ventilation between plants.
• Avoid excess humidity by using drip irrigation instead of overhead spraying.""",
    
    "Potato Common Scab": """• Lower soil pH below 5.5 using sulfur amendments.
• Use resistant potato varieties.
• Maintain soil moisture during tuber development.
• Use clean, disease-free planting material.""",
    
    "Potato Dry Rot": """• Handle tubers carefully during harvest to avoid injuries.
• Treat seed potatoes with fungicide before planting.
• Store harvested potatoes at optimal temperatures (3-4°C) and low humidity.
• Discard and destroy heavily infected tubers.""",
    
    "Potato Healthy": "• Maintain crop rotation, soil health, and timely disease monitoring.",
    
    "Potato Scab": """• Maintain consistent soil moisture during tuber formation.
• Use scab-resistant potato varieties.
• Avoid liming fields unless absolutely necessary.""",
    
    "Rice Leaf Blight": """• Use certified disease-free seeds and resistant varieties.
• Apply recommended bactericides like streptomycin if needed.
• Maintain proper field sanitation and remove infected debris.
• Avoid excessive nitrogen fertilizer application.""",
    
    "Rice Healthy": "• Maintain balanced fertilization, proper water levels, and pest monitoring.",
    
    "Rice Leaf Scald": """• Improve soil fertility with balanced fertilization (NPK).
• Grow resistant varieties if available.
• Ensure good drainage and avoid water stress.""",
    
    "Rice Sheath Blight": """• Use fungicides like azoxystrobin or validamycin at early symptoms.
• Maintain wider plant spacing to reduce humidity.
• Grow moderately resistant rice varieties.""",
    
    "Wheat Black Rust": """• Plant resistant wheat varieties.
• Apply fungicides like propiconazole at the first sign of infection.
• Destroy volunteer wheat plants that can host the fungus.""",
    
    "Wheat Head Blight": """• Apply fungicides such as tebuconazole during flowering stage.
• Rotate crops with non-host plants like soybeans.
• Plow debris from previous wheat crops into the soil to destroy overwintering fungus.""",
    
    "Wheat Healthy": "• Keep up with disease monitoring and maintain soil nutrition balance.",
    
    "Wheat Mildew": """• Use resistant wheat varieties.
• Apply systemic fungicides when early symptoms are spotted.
• Improve plant airflow by optimizing seeding rates.""",
    
    "Wheat Smut": """• Use fungicide-treated certified seeds.
• Remove infected heads before seed collection.
• Rotate crops regularly to break the infection cycle."""
}

# Folder path
folder_path = r'C:\Users\Param\Desktop\PRACTICUM\Model\Test'

# Loop through images
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(folder_path, filename)

        # Load and preprocess image
        img = image.load_img(img_path, target_size=img_size)
        img_array = image.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        img_array_expanded = img_array_expanded / 255.0  # Normalize

        # Predict
        prediction = model.predict(img_array_expanded)
        predicted_class_idx = np.argmax(prediction, axis=1)[0]

        print(f"\nProcessing {filename}")

        # Get predicted label, reason, and cure
        if predicted_class_idx < len(class_labels):
            predicted_label = class_labels[predicted_class_idx]
            reason = disease_reasons.get(predicted_label, "Reason unknown.")
            cure = cure_recommendations.get(predicted_label, "Cure information not available.")
        else:
            predicted_label = "Unknown Class"
            reason = "Reason unknown."
            cure = "Cure information not available."

        # Display image with prediction
        plt.imshow(np.array(img))
        plt.title(f"{filename}\nPredicted: {predicted_label}", fontsize=10)
        plt.axis('off')
        plt.show()

        # Display reason and cure
        print(f"Prediction: {predicted_label}")
        print(f"Reason: {reason}")
        print(f"Cure / Treatment:\n{cure}")
        print("-" * 80)
