# Plant Disease Detection

CNN image classification model for detecting plant diseases from leaf images using TensorFlow/Keras.

🔗 Project link: https://plant-disease-detection-ffdpakfvcdd4dqhm.eastasia-01.azurewebsites.net/ 

🗂️ Model link: https://drive.google.com/file/d/10ilNbvfSOxj8TYUHupQAAWzCBkuOe-bQ/view?usp=sharing 

---

## Overview

This dataset contains high-resolution images of infected and healthy plant leaves, categorized into **23 distinct classes**.

The primary goal is to enable machine learning and deep learning models to accurately detect and classify plant diseases across five major crops:

* Apple
* Corn (Maize)
* Pepper (Bell)
* Potato
* Tomato

---

## Dataset Structure

Each image is stored in a folder named after its class label using the following naming convention:

```text
<PlantName>___<DiseaseName>
```

or

```text
<PlantName>___healthy
```

if the leaf is healthy.

### Examples

* `Tomato_Early_blight` → Tomato leaves infected with Early Blight
* `Apple___healthy` → Healthy apple leaves

---

## Class Labels (23 Total)

🍎 Apple

* Apple___Apple_scab
* Apple___Black_rot
* Apple___Cedar_apple_rust
* Apple___healthy

---

🌽 Corn (Maize)

* Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot
* Corn_(maize)__*Common_rust*
* Corn_(maize)___Northern_Leaf_Blight
* Corn_(maize)___healthy

---

🫑 Pepper (Bell)

* Pepper__bell___Bacterial_spot
* Pepper__bell___healthy

---

🥔 Potato

* Potato___Early_blight
* Potato___Late_blight
* Potato___healthy

---

🍅 Tomato

* Tomato_Bacterial_spot
* Tomato_Early_blight
* Tomato_Late_blight
* Tomato_Leaf_Mold
* Tomato_Septoria_leaf_spot
* Tomato_Spider_mites_Two_spotted_spider_mite
* Tomato__Target_Spot
* Tomato__Tomato_YellowLeaf__Curl_Virus
* Tomato__Tomato_mosaic_virus
* Tomato_healthy

---

### Notes

* Ensure balanced train-validation splitting for all 23 classes.
* Some classes may contain fewer samples than others.
* Use `shuffle=False` during evaluation to avoid label mismatch issues.

---

