import streamlit as st
import ollama
import io
import base64
import pandas as pd
from PIL import Image
from datetime import datetime
import csv
import json
import tempfile
import os

# Add PDF support
try:
    from pdf2image import convert_from_bytes
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    st.warning("PDF support requires the pdf2image library. Install it with: pip install pdf2image")

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
    st.header("Upload Files")
    # File uploader for multiple images and PDFs
    uploaded_files = st.file_uploader(
        "Choose images or PDFs", 
        accept_multiple_files=True, 
        type=['png', 'jpg', 'jpeg', 'pdf']
    )
    
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} files")
        
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
            
            # Option to process PDF pages separately or as a whole
            if any(file.name.lower().endswith('.pdf') for file in uploaded_files):
                pdf_process_mode = st.radio(
                    "How to process PDF files:",
                    ["Process each page separately", "Process entire PDF as one document"]
                )
            else:
                pdf_process_mode = "Process each page separately"
        
        # Process button in sidebar
        process_button = st.button("Process Files")
    else:
        st.info("Please upload images or PDF files to analyze")
        process_button = False
        extraction_mode = "General description"
        pdf_process_mode = "Process each page separately"

# Function to process an image with Ollama
def process_image(image, filename, fields=None):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    if fields is None:
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
        
        result = {
            'filename': filename,
            'description': response['message']['content']
        }
        return result, response['message']['content'], None
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
        
        content = response['message']['content']
        
        # Store the raw response
        result = {
            'filename': filename,
            'extraction': content
        }
        
        # Try to extract structured data
        structured_data = {'filename': filename}
        
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
            
        return result, content, structured_data

# Main area for results
if uploaded_files and process_button:
    st.header("Processing Results")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Clear previous results when starting a new batch
    st.session_state.results = []
    st.session_state.structured_results = []
    
    # Prepare fields if using custom extraction
    if extraction_mode == "Custom field extraction":
        fields = [field.strip() for field in custom_fields.split(",")]
    else:
        fields = None
    
    # Count total items to process (including PDF pages)
    total_items = 0
    pdf_page_counts = {}
    
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)  # Reset file pointer after reading
        
        if uploaded_file.name.lower().endswith('.pdf') and PDF_SUPPORT:
            if pdf_process_mode == "Process each page separately":
                # Count pages
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(file_bytes)
                        temp_file_path = temp_file.name
                    
                    images = convert_from_bytes(file_bytes)
                    pdf_page_counts[uploaded_file.name] = len(images)
                    total_items += len(images)
                    
                    os.unlink(temp_file_path)
                except Exception as e:
                    st.error(f"Error processing PDF {uploaded_file.name}: {e}")
                    total_items += 1  # Count as one item even on error
            else:
                total_items += 1  # Count whole PDF as one item
        else:
            total_items += 1  # Regular image or unsupported PDF
    
    processed_count = 0
    
    # Process each file
    for i, uploaded_file in enumerate(uploaded_files):
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)  # Reset file pointer after reading
        
        # Handle PDF files
        if uploaded_file.name.lower().endswith('.pdf'):
            if not PDF_SUPPORT:
                st.error(f"Cannot process PDF file {uploaded_file.name}. Please install pdf2image library.")
                processed_count += 1
                progress_bar.progress(processed_count / total_items)
                continue
                
            try:
                # Convert PDF to images
                images = convert_from_bytes(file_bytes)
                
                if pdf_process_mode == "Process each page separately":
                    # Process each page as a separate image
                    for j, image in enumerate(images):
                        page_filename = f"{uploaded_file.name} (Page {j+1})"
                        status_text.text(f"Processing {page_filename}")
                        
                        result, content, structured_data = process_image(image, page_filename, fields)
                        st.session_state.results.append(result)
                        
                        if structured_data:
                            st.session_state.structured_results.append(structured_data)
                        
                        # Display the processed image and its results
                        st.subheader(page_filename)
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.image(image, width=250)
                        with col2:
                            st.write(content)
                            if structured_data and len(structured_data) > 1:
                                st.success("Successfully extracted structured data")
                                st.json(structured_data)
                        
                        st.divider()
                        
                        processed_count += 1
                        progress_bar.progress(processed_count / total_items)
                else:
                    # Process the entire PDF as one document (using the first page as the image)
                    status_text.text(f"Processing {uploaded_file.name} (entire document)")
                    
                    # Use first page as the image
                    result, content, structured_data = process_image(images[0], uploaded_file.name, fields)
                    st.session_state.results.append(result)
                    
                    if structured_data:
                        st.session_state.structured_results.append(structured_data)
                    
                    # Display the first page and results
                    st.subheader(f"{uploaded_file.name} (full document)")
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(images[0], width=250)
                        if len(images) > 1:
                            st.info(f"PDF has {len(images)} pages. Showing first page only.")
                    with col2:
                        st.write(content)
                        if structured_data and len(structured_data) > 1:
                            st.success("Successfully extracted structured data")
                            st.json(structured_data)
                    
                    st.divider()
                    
                    processed_count += 1
                    progress_bar.progress(processed_count / total_items)
                    
            except Exception as e:
                st.error(f"Error processing PDF {uploaded_file.name}: {e}")
                processed_count += 1
                progress_bar.progress(processed_count / total_items)
        
        else:
            # Process regular image file
            status_text.text(f"Processing image {uploaded_file.name}")
            
            try:
                image = Image.open(uploaded_file)
                
                result, content, structured_data = process_image(image, uploaded_file.name, fields)
                st.session_state.results.append(result)
                
                if structured_data:
                    st.session_state.structured_results.append(structured_data)
                
                # Display the processed image and its results
                st.subheader(f"Image: {uploaded_file.name}")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(image, width=250)
                with col2:
                    st.write(content)
                    if structured_data and len(structured_data) > 1:
                        st.success("Successfully extracted structured data")
                        st.json(structured_data)
                
                st.divider()
                
            except Exception as e:
                st.error(f"Error processing image {uploaded_file.name}: {e}")
            
            processed_count += 1
            progress_bar.progress(processed_count / total_items)
    
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
            csv_writer.writerow([result['filename'], result.get('description', result.get('extraction', ''))])
        
        # Generate a filename with current date/time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"image_analysis_{timestamp}.csv"
        
        st.success("All files have been processed successfully!")
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

# Display instructions when no files are processed yet
if not uploaded_files:
    st.info("👈 Upload files using the sidebar to get started")
    st.write("""
    ## How to use this app:
    1. Upload one or more images or PDF files using the sidebar on the left
    2. Choose between general description or custom field extraction
    3. If using custom extraction, specify the fields you want to extract (e.g., Invoice number, Date, Total amount)
    4. For PDFs, choose whether to process each page separately or the entire document at once
    5. Click the 'Process Files' button to analyze them
    6. View the results for each image or PDF page
    7. Download results as a CSV file
    
    This app uses the Gemma 3 12B vision model to analyze and describe images and PDFs.
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
