import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="COVID-19 AI Detector",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("COVID-19 AI Dashboard")

st.sidebar.write(
    "Deep Learning based Chest X-ray Classification"
)

st.sidebar.write("---")

st.sidebar.info(
    """
    Classes:
    
    • COVID19
    
    • NORMAL
    
    • PNEUMONIA
    """
)

# ============================================================
# LOAD MODEL
# ============================================================

model = tf.keras.models.load_model(
    'best_augmentation_model.keras'
)

# ============================================================
# CLASS NAMES
# ============================================================

class_names = [
    'COVID19',
    'NORMAL',
    'PNEUMONIA'
]

# ============================================================
# TITLE
# ============================================================

st.markdown(
    """
    <h1 style='text-align: center; color: #00FFAA;'>
    COVID-19 Detection using Deep Learning
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h4 style='text-align: center; color: gray;'>
    Chest X-ray Classification using ResNet50
    </h4>
    """,
    unsafe_allow_html=True
)

st.write("---")

# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=['jpg', 'jpeg', 'png']
)

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image):

    image = image.convert('RGB')

    image = image.resize((224, 224))

    image_array = np.asarray(image)

    image_array = image_array.astype(np.float32) / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    prediction = model.predict(
        image_array,
        verbose=0
    )

    predicted_class = np.argmax(
        prediction[0]
    )

    confidence = float(
        np.max(prediction[0])
    )

    return (
        class_names[predicted_class],
        confidence,
        prediction[0]
    )

# ============================================================
# MAIN SECTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    # ========================================================
    # IMAGE DISPLAY
    # ========================================================

    with col1:

        st.image(
            image,
            caption='Uploaded Chest X-ray',
            use_container_width=True
        )

    # ========================================================
    # PREDICTION OUTPUT
    # ========================================================

    with col2:

        if st.button("Predict"):

            with st.spinner("Analyzing Chest X-ray..."):

                prediction, confidence, prediction_probs = predict_image(image)

            # ====================================================
            # PREDICTION RESULT
            # ====================================================

            if prediction == 'COVID19':

                st.error(
                    f"Prediction : {prediction}"
                )

            elif prediction == 'NORMAL':

                st.success(
                    f"Prediction : {prediction}"
                )

            else:

                st.warning(
                    f"Prediction : {prediction}"
                )

            # ====================================================
            # CONFIDENCE SCORE
            # ====================================================

            st.metric(
                label="Confidence Score",
                value=f"{confidence:.2f}"
            )

            # ====================================================
            # PROBABILITY TABLE
            # ====================================================

            prob_df = pd.DataFrame({
                'Class': class_names,
                'Probability': prediction_probs
            })

            st.write("### Prediction Probabilities")

            st.table(prob_df)

            # ====================================================
            # BAR CHART
            # ====================================================

            st.write("### Probability Distribution")

            chart_df = pd.DataFrame(
                {
                    'Probability': prediction_probs
                },
                index=class_names
            )

            st.bar_chart(chart_df)

# ============================================================
# MODEL DETAILS
# ============================================================

with st.expander("Model Details"):

    st.write("Model : ResNet50")

    st.write("Framework : TensorFlow / Keras")

    st.write("Image Size : 224 x 224")

    st.write("Classes : COVID19 / NORMAL / PNEUMONIA")

    st.write("Transfer Learning + Data Augmentation")

# ============================================================
# FOOTER
# ============================================================

st.write("---")

st.caption(
    "Developed using Streamlit, TensorFlow and ResNet50"
)