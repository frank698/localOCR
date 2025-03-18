import streamlit as st
import ollama
import io
import base64
import pandas as pd
from PIL import Image
from datetime import datetime
import csv
import json

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Curiosity AI Scans",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Display the app title
st.title("Curiosity AI Scans")

# Initialize session state for storing results if it doesn't exist
if 'results' not in st.session_state:
    st.session_state.results = []
if 'structured_results' not in st.session_state:
    st.session_state.structured_results = []

# Create a sidebar for the file upload functionality
with st.sidebar:
    st.header("Upload Images")
    # File uploader for multiple images
    uploaded_files = st.file_uploader("Choose images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
    
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} images")
        
        # Add option for structured data extraction
        st.header("Data Extraction Options")
        extraction_mode = st.radio(
            "Choose extraction mode:",
            ["General description", "Custom field extraction"]
        )
        
        # If custom extraction is selected, show field input
        if extraction_mode == "Custom field extraction":
            st.write("Enter the fields you want to extract (separated by commas):")
            custom_fields = st.text_area(
                "Example: Invoice number, Date, Company name, Total amount", 
                value="Invoice number, Date, Company name, Total amount"
            )
        
        # Process button in sidebar
        process_button = st.button("Process Images")
    else:
        st.info("Please upload one or more images to analyze")
        process_button = False
        extraction_mode = "General description"

# Main area for results
if uploaded_files and process_button:
    st.header("Processing Results")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Clear previous results when starting a new batch
    st.session_state.results = []
    st.session_state.structured_results = []
    
    # Get fields if using custom extraction
    if extraction_mode == "Custom field extraction":
        fields = [field.strip() for field in custom_fields.split(",")]
        # Create a list to store structured results
        st.session_state.structured_results = []
    
    # Process each image
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing image {i+1}/{len(uploaded_files)}")
        
        # Process the image with Ollama
        image = Image.open(uploaded_file)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        if extraction_mode == "General description":
            # General description mode
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
            
            # Store general results
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
        
        else:
            # Custom field extraction mode
            fields_str = ", ".join(fields)
            prompt = f"Extract the following information from this image: {fields_str}. Return the results in JSON format with these exact field names."
            
            response = ollama.chat(
                model='gemma3:12b',
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [base64.b64encode(img_byte_arr).decode('utf-8')]
                    }
                ]
            )
            
            # Extract JSON data from response
            content = response['message']['content']
            
            # Store the raw response
            result = {
                'filename': uploaded_file.name,
                'extraction': content
            }
            st.session_state.results.append(result)
            
            # Try to extract structured data
            structured_data = {'filename': uploaded_file.name}
            
            # Look for JSON in the response
            try:
                # Try to find JSON content between ```json and ``` markers
                if "```json" in content and "```" in content.split("```json")[1]:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                    extracted_data = json.loads(json_str)
                    structured_data.update(extracted_data)
                else:
                    # Try to find any JSON object in the text
                    json_str = content
                    for field in fields:
                        field_clean = field.strip()
                        if f'"{field_clean}"' in json_str or f"'{field_clean}'" in json_str:
                            try:
                                extracted_data = json.loads(json_str)
                                structured_data.update(extracted_data)
                                break
                            except:
                                pass
            except:
                # If JSON parsing fails, we'll keep the raw text
                pass
                
            # Add to structured results
            st.session_state.structured_results.append(structured_data)
            
            # Display the processed image and extraction results
            st.subheader(f"Image {i+1}: {uploaded_file.name}")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image, width=250)
            with col2:
                st.write(content)
                
                # Show parsed fields if available
                if len(structured_data) > 1:  # More than just filename
                    st.success("Successfully extracted structured data")
                    st.json(structured_data)
                else:
                    st.warning("Could not parse structured data, using raw extraction")
        
        st.divider()  # Add a divider between results
        
        # Update progress
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text("Processing complete!")
    
# Display download buttons in main area if we have results
if st.session_state.results:
    st.header("Download Results")
    
    # Create general CSV data
    csv_data = io.StringIO()
    csv_writer = csv.writer(csv_data)
    
    if extraction_mode == "General description" or len(st.session_state.structured_results) == 0:
        # Original CSV format with descriptions
        csv_writer.writerow(['Filename', 'Description'])
        for result in st.session_state.results:
            csv_writer.writerow([result['filename'], result['description'] if 'description' in result else result['extraction']])
        
        # Generate a filename with current date/time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"image_analysis_{timestamp}.csv"
        
        st.success("All images have been processed successfully!")
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data.getvalue(),
            file_name=csv_filename,
            mime="text/csv",
            use_container_width=True
        )
    
    # Create structured CSV if available
    if extraction_mode == "Custom field extraction" and len(st.session_state.structured_results) > 0:
        # Get all possible fields from the results
        all_fields = set(['filename'])
        for result in st.session_state.structured_results:
            all_fields.update(result.keys())
        
        # Convert to list and sort
        field_list = sorted(list(all_fields))
        
        # Create structured CSV
        structured_csv = io.StringIO()
        structured_writer = csv.writer(structured_csv)
        structured_writer.writerow(field_list)
        
        for result in st.session_state.structured_results:
            row = [result.get(field, '') for field in field_list]
            structured_writer.writerow(row)
        
        # Generate a filename with current date/time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        structured_filename = f"structured_data_{timestamp}.csv"
        
        st.success("Structured data extracted successfully!")
        st.download_button(
            label="📥 Download Structured Data as CSV",
            data=structured_csv.getvalue(),
            file_name=structured_filename,
            mime="text/csv",
            use_container_width=True
        )

# Display instructions when no images are processed yet
if not uploaded_files:
    st.info("👈 Upload images using the sidebar to get started")
    st.write("""
    ## How to use this app:
    1. Upload one or more images using the sidebar on the left
    2. Choose between general description or custom field extraction
    3. If using custom extraction, specify the fields you want to extract (e.g., Invoice number, Date, Total amount)
    4. Click the 'Process Images' button to analyze them
    5. View the results for each image
    6. Download results as a CSV file
    
    This app uses the Gemma 3 12B vision model to analyze and describe images.
    """)

# Add a footer with attribution
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; margin-top: 20px; opacity: 0.7;">
        Made with ❤️ by Adrian with Claude - <a href="https://ad1x.com" target="_blank">ad1x.com</a>
    </div>
    """, 
    unsafe_allow_html=True
)
