import streamlit as st
import ollama
from PIL import Image
import io
import base64
import pandas as pd
from datetime import datetime
import csv

# Page configuration
st.set_page_config(
    page_title="Gemma-3 OCR",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description in main area
st.markdown("""
    # <img src="data:image/png;base64,{}" width="50" style="vertical-align: -12px;"> Gemma-3 OCR
""".format(base64.b64encode(open("./assets/gemma3.png", "rb").read()).decode()), unsafe_allow_html=True)

# Add clear button to top right
col1, col2 = st.columns([6,1])
with col2:
    if st.button("Clear 🗑️"):
        if 'ocr_result' in st.session_state:
            del st.session_state['ocr_result']
        st.rerun()

st.markdown('<p style="margin-top: -20px;">Extract structured text from images using Gemma-3 Vision!</p>', unsafe_allow_html=True)
st.markdown("---")

# Initialize session state for storing results if it doesn't exist
if 'results' not in st.session_state:
    st.session_state.results = []

# Create a sidebar for the file upload functionality
with st.sidebar:
    st.header("Upload Images")
    # File uploader for multiple images
    uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
    
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} images")
        # Process button in sidebar
        process_button = st.button("Process Images")
    else:
        st.info("Please upload one or more images to analyze")
        process_button = False

# Main area for results
if uploaded_files and process_button:
    st.header("Processing Results")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Clear previous results when starting a new batch
    st.session_state.results = []
    
    # Process each image
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing image {i+1}/{len(uploaded_files)}")
        
        # Process the image with Ollama
        image = Image.open(uploaded_file)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        response = ollama.chat(
            model='gemma3:12b',
            messages=[
                {
                    'role': 'user',
                    'content': 'Describe what you see in this image in detail.',
                    'images': [base64.b64encode(img_byte_arr).decode('utf-8')]
                }
            ]
        )
        
        # Store results
        result = {
            'filename': uploaded_file.name,
            'description': response['message']['content']
        }
        st.session_state.results.append(result)
        
        # Display the processed image and its description in the main area
        st.subheader(f"Image {i+1}: {uploaded_file.name}")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, width=250)
        with col2:
            st.write(response['message']['content'])
        
        st.divider()  # Add a divider between results
        
        # Update progress
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text("Processing complete!")
    
# Display download button in main area if we have results
if st.session_state.results:
    st.header("Download Results")
    
    # Create CSV data
    csv_data = io.StringIO()
    csv_writer = csv.writer(csv_data)
    csv_writer.writerow(['Filename', 'Description'])
    for result in st.session_state.results:
        csv_writer.writerow([result['filename'], result['description']])
    
    # Generate a filename with current date/time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"image_analysis_{timestamp}.csv"
    
    # Create download button in a prominent location
    st.success("All images have been processed successfully!")
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_data.getvalue(),
        file_name=csv_filename,
        mime="text/csv",
        use_container_width=True
    )

# Display instructions when no images are processed yet
if not uploaded_files:
    st.info("👈 Upload images using the sidebar to get started")
    st.write("""
    ## How to use this app:
    1. Upload one or more images using the sidebar on the left
    2. Click the 'Process Images' button to analyze them
    3. View the results for each image
    4. Download all results as a CSV file
    """)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Gemma-3 Vision Model")
