import streamlit as st
import tensorflow as tf
import numpy as np 
from PIL import Image 

#Tensorflow Model Prediction 
from io import BytesIO

def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = Image.open(BytesIO(test_image.read()))
    image = image.resize((128, 128))  # Resize the image
    input_arr = np.array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Expand the dimensions to create batch dimension
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index


#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Home Page 
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path="home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    

**Welcome to the Plant Disease Recognition System! 🌿🔍**

Our goal is to aid in the efficient identification of plant diseases. By simply uploading an image of a plant, our system will analyze it to detect any signs of diseases. Together, let's safeguard our crops and ensure a bountiful harvest!

### How It Works
1. **Upload Image:** Navigate to the **Disease Recognition** page and upload an image of a plant suspected of having diseases.
2. **Analysis:** Our system employs advanced algorithms to process the image, identifying potential diseases.
3. **Results:** Explore the results and receive recommendations for further action.

### Why Choose Us?
- **Accuracy:** Leveraging state-of-the-art machine learning techniques ensures precise disease detection.
- **User-Friendly:** Enjoy a simple and intuitive interface for a seamless user experience.
- **Fast and Efficient:** Receive results within seconds, facilitating prompt decision-making.

### Get Started
Head over to the **Disease Recognition** page in the sidebar to upload an image and witness the effectiveness of our Plant Disease Recognition System!

### About Us
Discover more about the project, our team, and our objectives on the **About** page.

---

Feel free to tweak anything further if you have specific preferences!
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))
