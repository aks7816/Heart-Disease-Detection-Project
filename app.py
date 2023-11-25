# Step 1: install Gradio
#!pip install --quiet gradio

import gradio as gr
import numpy as np
import tensorflow as tf
import cv2

input1 = gr.Textbox(label="Input Features. Please enter them in array form. Please use this template: [age (integer), sex (binary), chest pain type (categorical with 4 levels), resting blood pressure (continuous), serum cholestrol (continuous), fasting blood sugar (binary), resting electrocardiographic results (categorical with 3 levels), max heart rate (continuous), exercise induced angina (binary), oldpeak, slope of the peak exercise ST segment (categorical with 3 levels), number of major vessels colored by flourosopy (categorical with 4 levels), thal(3 = normal; 6 = fixed defect; 7 = reversable defect)] ", lines=10)

input2_dropdown = gr.Dropdown(choices=["Logistic Regression", "Support Vector Machine","Linear Discriminant Analysis"], label="Method")

import joblib

filename = 'log_model.joblib'
log_model = joblib.load(filename)

filename = 'svm_model.joblib'
svm_model = joblib.load(filename)

filename = 'lda_model.joblib'
lda_model = joblib.load(filename)



def predict(input1, input2):
    try:
        # Convert the input string to a list of floats
        newInput = [float(val) for val in input1.strip("[]").split(",")]
        newInput = np.array(newInput).reshape(1, -1)

        # Select the appropriate model
        if input2 == "Logistic Regression":
            preds = log_model.predict(newInput)
        elif input2 == "Support Vector Machine":
            preds = svm_model.predict(newInput)
        elif input2 == "Linear Discriminant Analysis":
            preds = lda_model.predict(newInput)
        else:
            preds = None

        if preds is not None:
            if preds[0] == 1:
                output_label = "Absence of Heart Disease"
            elif preds[0] == 2:
                output_label = "Presence of Heart Disease"
        else:
            output_label = " "

        return output_label
    except Exception as e:
       return " "#f"Error: {str(e)}"

label = gr.Label(label="Diagnosis")

interface = gr.Interface(
    fn=predict,
    inputs=[input1, input2_dropdown],
    examples=[["[70.0, 1.0, 4.0, 130.0, 322.0, 0.0, 2.0, 109.0, 0.0, 2.4, 2.0, 3.0, 3.0]"]],
    outputs=label
).launch(debug=True, share=True)
