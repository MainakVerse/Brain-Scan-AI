import os
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from src.model import MyModel, load_model
from src.utils import predict
import base64
import google.generativeai as genai


GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

def get_gemini_response(prompt):
    """Fetch real-time response from Gemini API."""
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text if response else "I'm unable to provide a response at the moment."


# Custom CSS for the application
def load_css():
    st.markdown("""
    <style>
        /* Main Theme Colors */
        :root {
            --primary: #3E4E88;
            --secondary: #6D8BC9;
            --accent: #C79F27;
            --background: #60759f;
            --text-dark: #333340;
            --text-light: #ffffff;
            --container-bg: #E1E5EE;
            --sidebar-bg: #000000;
        }
        
        /* General Styling */
        .stApp {
            background-color: var(--background);
            color: var(--text-dark);
        }
        
        /* Main Header */
        h1 {
            color: var(--primary);
            padding-bottom: 10px;
            border-bottom: 2px solid var(--accent);
            margin-bottom: 20px;
            font-weight: 600;
        }
        
        /* Sidebar */
        .sidebar .sidebar-content {
            background-color: var(--sidebar-bg);
            color: var(--text-light);
            padding-top: 20px;
        }
        
        .sidebar h2 {
            color: var(--text-light);
            font-size: 1.5rem;
            margin-bottom: 20px;
            border-bottom: 1px solid var(--accent);
            padding-bottom: 10px;
        }
        
        .sidebar .current-tab {
            background-color: var(--accent);
            padding: 10px;
            border-radius: 5px;
            margin-top: 20px;
            font-weight: bold;
        }
        
        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: var(--container-bg);
            border-radius: 4px 4px 0 0;
            padding: 10px 20px;
            color: var(--text-dark);
        }
        
        .stTabs [aria-selected="true"] {
            background-color: var(--primary) !important;
            color: var(--text-light) !important;
        }
        
        /* Container Styling */
        .content-container {
            background-color: var(--container-bg);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        
        /* Button Styling */
        .stButton > button {
            background-color: var(--primary);
            color: var(--text-light);
            border: none;
            padding: 10px 15px;
            border-radius: 5px;
            font-weight: bold;
            transition: all 0.3s;
        }
        
        .stButton > button:hover {
            background-color: var(--secondary);
            color: var(--text-light);
        }
        
        /* File Uploader */
        .uploadedFileData {
            border: 1px dashed var(--accent);
            border-radius: 5px;
            padding: 10px;
        }
        
        /* Prediction Styling */
        .prediction-result {
            background-color: var(--primary);
            color: var(--text-light);
            padding: 15px;
            border-radius: 8px;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
            font-size: 1.5rem;
        }
        
        /* Sample Images Grid */
        .sample-image {
            border: 2px solid var(--primary);
            border-radius: 5px;
            transition: transform 0.3s;
        }
        
        .sample-image:hover {
            transform: scale(1.05);
        }
        
        /* Sample Info Box */
        .sample-info-box {
            background-color: var(--container-bg);
            border-left: 4px solid var(--accent);
            padding: 15px;
            border-radius: 0 5px 5px 0;
            height: 100%;
        }
        
        /* Chatbot UI */
        .chat-message {
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        
        .user-message {
            background-color: var(--secondary);
            color: var(--text-light);
            margin-left: 20px;
            border-radius: 10px 10px 0 10px;
        }
        
        .bot-message {
            background-color: var(--container-bg);
            color: var(--text-dark);
            margin-right: 20px;
            border-radius: 10px 10px 10px 0;
        }
        
        .chat-input {
            background-color: var(--container-bg);
            border: 1px solid var(--primary);
            border-radius: 5px;
            padding: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def get_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join("models", "model_38")
    return load_model(model_path, device), device

model, device = get_model()

# Define the transformation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# map labels from int to string
label_dict = {
    0: "No Tumor",
    1: "Pituitary",
    2: "Glioma",
    3: "Meningioma",
    4: "Other",
}

# Sample information for each class
sample_info = {
    "Glioma": "Gliomas are tumors that grow from glial cells in the brain and spinal cord. They are the most common type of brain tumor, accounting for about 30% of all brain and central nervous system tumors. Symptoms may include headaches, seizures, and cognitive changes.",

    "Meningioma": "Meningiomas arise from the meninges, the membranes that surround the brain and spinal cord. They represent about 37% of primary brain tumors and are usually slow-growing and benign. Common symptoms include headaches, vision changes, and seizures.",

    "Pituitary": "Pituitary tumors develop in the pituitary gland at the base of the brain. They account for about 10-15% of all brain tumors. Symptoms may include hormone imbalances, vision problems, and headaches.",

    "No Tumor": "This classification indicates a normal brain scan with no detectable tumor present. Regular monitoring may still be recommended based on clinical symptoms.",

    "Other": "This category includes less common brain tumors such as ependymomas, craniopharyngiomas, and lymphomas. Symptoms vary widely depending on the specific type and location."
}

# Process image got from user before passing to the model
def preprocess_image(image):
    preprocessed_image = transform(image).unsqueeze(0)
    return preprocessed_image

# Function to create a basic mock chatbot
def chatbot_interface():
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.subheader("Medical Consultation Assistant")
    st.write("Use this chat to ask questions about brain tumors and MRI results. Our AI assistant will try to provide helpful information.")
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        role, content = message['role'], message['content']
        if role == 'user':
            st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message"><strong>Assistant:</strong> {content}</div>', unsafe_allow_html=True)
    
    # Chat input
    user_input = st.text_input("Type your question here:", key="chat_input", placeholder="E.g., What are common symptoms of a meningioma?")
    
    if st.button("Send", key="send_button"):
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Get real-time response from Gemini API
            response = get_gemini_response(user_input)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Rerun to update the chat display
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Function to display sample MRI images
def display_samples():
    st.subheader("Sample MRI Images by Classification")

    sample_images = {
        "Glioma": "images/glioma.jpg",
        "Meningioma": "images/meningioma.jpg",
        "Pituitary": "images/pituitary.jpg",
        "No Tumor": "images/no_tumor.jpg",
        "Other": "images/other.png"
    }
    
    for tumor_type, image_path in sample_images.items():
        st.markdown(f'<div class="content-container">', unsafe_allow_html=True)
        cols = st.columns([1, 2])
        
        with cols[0]:
            # Load and display the specific MRI image for each tumor type
            if os.path.exists(image_path):
                st.image(image_path, caption=f"{tumor_type} MRI Sample", use_column_width=True)
            else:
                st.image("https://via.placeholder.com/300x300.png?text=Image+Not+Available", 
                         caption=f"{tumor_type} MRI Sample", use_column_width=True)
            
        with cols[1]:
            st.markdown(f'<div class="sample-info-box">', unsafe_allow_html=True)
            st.markdown(f"### {tumor_type}")
            st.write(sample_info.get(tumor_type, "No additional information available."))
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# About section content
def about_section():
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.subheader("About Brain Tumor Classification System")
    
    st.write("""
    This application uses deep learning to analyze brain MRI images and classify them into different categories of brain tumors. The system is designed to assist medical professionals in the diagnostic process.
    """)
    
    st.markdown("### How It Works")
    st.write("""
    The classification system employs a convolutional neural network (CNN) trained on thousands of brain MRI images. The model has learned to identify patterns and features associated with different types of brain tumors.
    
    When you upload an MRI image, the system:
    1. Preprocesses the image to match the format expected by the model
    2. Passes the processed image through the neural network
    3. Analyzes the output to determine the most likely classification
    4. Presents the results with the associated confidence level
    """)
    
    st.markdown("### Classification Categories")
    st.write("""
    The system can identify the following categories:
    - **Glioma**: Tumors that arise from glial cells
    - **Meningioma**: Tumors that develop in the meninges
    - **Pituitary**: Tumors of the pituitary gland
    - **Other**: Less common brain tumor types
    - **No Tumor**: Normal brain MRI with no detectable tumor
    """)
    
    st.markdown("### Limitations")
    st.write("""
    While this tool can assist in the diagnostic process, it should not replace clinical judgment. The final diagnosis should always be made by qualified healthcare professionals considering all available clinical information.
    """)
    
    st.markdown("### Development Team")
    st.write("""
    This application was developed by a team of medical professionals, data scientists, and software engineers dedicated to improving diagnostic tools through artificial intelligence.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Main detection tab functionality
def detection_tab():
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    st.subheader("Brain Tumor Detection")
    st.write("Upload a brain MRI image below for classification.")
    
    # Image upload section
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            
            # Display uploaded image
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image, caption="Uploaded MRI Image", width=250)
            
            # Process and predict
            with st.spinner('Analyzing image...'):
                # Preprocess the image
                preprocessed_image = preprocess_image(image).to(device)
                # Make prediction
                predicted_class = predict(model, preprocessed_image, device)
                
                # Display prediction
                with col2:
                    st.markdown(f'<div class="prediction-result">Prediction: {label_dict[predicted_class]}</div>', unsafe_allow_html=True)
                    st.markdown(f"### About {label_dict[predicted_class]}")
                    st.write(sample_info.get(label_dict[predicted_class], "No information available for this classification."))
        
        except Exception as e:
            st.error(f"Error processing image: {e}")
            st.write("Please upload a valid MRI image file.")
    
    st.markdown("""
    ### Instructions:
    1. Click the 'Browse files' button above
    2. Select a brain MRI image from your computer
    3. Wait for the system to analyze and classify the image
    4. Review the results shown on the right
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main app function
def main():
    # Apply custom CSS
    load_css()
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/12131/12131243.png", width=150)
        st.markdown('<h2>BrainScan AI</h2>', unsafe_allow_html=True)
        st.write('<p style="color:white";>Advanced brain tumor classification system using deep learning</p>', unsafe_allow_html=True)
        
        
        st.markdown("---")
        st.write('<p style="color:white";>© 2025 BrainScan AI</p>',unsafe_allow_html=True)
        st.write('<p style="color:yellow";>Made with 💙 by Mainak</p>',unsafe_allow_html=True)
        st.write(f'<p style="color: green";>Running on: {device}</p>',unsafe_allow_html=True)
    
    # Main content - Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Detection", "About", "Samples", "Consult"])
    
    with tab1:
        st.session_state.current_tab = "Detection"
        detection_tab()
    
    with tab2:
        st.session_state.current_tab = "About"
        about_section()
    
    with tab3:
        st.session_state.current_tab = "Samples"
        display_samples()
    
    with tab4:
        st.session_state.current_tab = "Consult"
        chatbot_interface()

if __name__ == "__main__":
    main()
