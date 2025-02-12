import streamlit as st
import cv2
import torch
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# ✅ Load models once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.write(f"🚀 Running on: **{device}**")

try:
    yolo_model = YOLO("yolov8s.pt").to(device)
    vgg16_model = tf.keras.models.load_model("vgg16_model.h5")
    st.sidebar.success("✅ Models loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Error loading models: {e}")
    st.stop()

# ✅ Class labels
class_labels = {
    0: '5_miles_speedlimit', 1: 'No_Cars Allowed', 2: 'No_Left_Turn',
    3: 'No_entry', 4: 'No_horn', 5: 'No_right_turn', 6: 'speedlimit_15'
}

def preprocess_for_vgg16(cropped_img):
    """ Preprocess image for VGG16 classification """
    cropped_img = cv2.resize(cropped_img, (224, 224))
    cropped_img = img_to_array(cropped_img)
    cropped_img = np.expand_dims(cropped_img, axis=0)
    cropped_img = preprocess_input(cropped_img)
    return cropped_img

def detect_and_classify(image):
    """ Detect objects using YOLO & classify using VGG16 """
    results = yolo_model(image)

    if not results or len(results[0].boxes) == 0:
        return image, "⚠️ No objects detected!"

    detected_classes = []

    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cropped_obj = image[y1:y2, x1:x2]

            if cropped_obj.shape[0] > 0 and cropped_obj.shape[1] > 0:
                processed_obj = preprocess_for_vgg16(cropped_obj)

                try:
                    predictions = vgg16_model.predict(processed_obj, verbose=0)
                    class_index = np.argmax(predictions)
                    class_name = class_labels.get(class_index, "Unknown")
                    confidence = predictions[0][class_index]
                    detected_classes.append(f"{class_name} ({confidence:.2f})")

                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, f"{class_name} ({confidence:.2f})",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except Exception as e:
                    detected_classes.append(f"❌ Error: {e}")

    return image, ", ".join(detected_classes)

# ✅ Streamlit App
st.title("🚦 Traffic Sign Detector using YOLO & VGG16")
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # Run detection & classification
    processed_image, result_text = detect_and_classify(image)

    # Convert BGR to RGB
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
    
    st.image(processed_image, caption="🔍 Processed Image", use_column_width=True)
    st.success(result_text)
