# Brain-Scan-AI

![image](https://github.com/user-attachments/assets/07d38c69-1dd8-40e4-9f50-f0845d7e7178)
This application uses deep learning to analyze brain MRI images and classify them into different categories of brain tumors. The system is designed to assist medical professionals in the diagnostic process.

## How it Works
The classification system employs a convolutional neural network (CNN) trained on thousands of brain MRI images. The model has learned to identify patterns and features associated with different types of brain tumors. 

When you upload an MRI image, the system:

- Preprocesses the image to match the format expected by the model
- Passes the processed image through the neural network
- Analyzes the output to determine the most likely classification
- Presents the results with the associated confidence level

## Categories

- Glioma: Tumors that arise from glial cells
- Meningioma: Tumors that develop in the meninges
- Pituitary: Tumors of the pituitary gland
- Other: Less common brain tumor types
- No Tumor: Normal brain MRI with no detectable tumor
