## Thank you to everyone for starring my repo! I'll do my best to extend the functionality regularly and fix things if people find problems. 

# Curiosity AI Scans

A simple application that uses AI vision models to analyze and describe images and PDF documents. Upload multiple files, get detailed descriptions powered by Gemma 3, and export the results to CSV.

## What this application does

This application allows you to:

- Upload multiple images (JPG, PNG) and PDF documents
- Process these files using the Gemma 3 12B vision model through Ollama
- Get detailed AI-generated descriptions of the content in each image or PDF page
- Extract specific fields from documents (e.g., invoice numbers, dates, amounts)
- View the analysis results in a clean, organized interface
- Export all results to a CSV file for further analysis or record-keeping

The app uses Streamlit for the interface, Ollama as the backend for running the Gemma 3 vision model locally, PIL for image processing, and PyMuPDF for PDF support.

## Installation and setup

### Step 1: Install Ollama

#### Linux
```bash
# setup ollama on linux 
curl -fsSL https://ollama.com/install.sh | sh
```

#### macOS
```bash
# Install Ollama using Homebrew
brew install ollama
# Or download from https://ollama.com/download
```

#### Windows
1. Download the Ollama installer from [https://ollama.com/download](https://ollama.com/download)
2. Run the installer and follow the on-screen instructions

### Step 2: Pull the Gemma-3 Vision model

After installing Ollama, pull the required model:
```bash
# On all platforms
ollama run gemma3:12b
```

### Step 3: Set up Python environment

Ensure you have Python 3.9 or later installed (Python 3.9-3.12 recommended for best compatibility).

#### Create and activate a virtual environment

##### Linux/macOS
```bash
# Clone or download this repository
git clone <repository-url>
cd <repository-folder>

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

##### Windows
```powershell
# Clone or download this repository
git clone <repository-url>
cd <repository-folder>

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate
```

### Step 4: Install dependencies

With your virtual environment activated, install the required packages:
```bash
# Install dependencies using requirements.txt
pip install -r requirements.txt

# If PyMuPDF is not included in requirements.txt, install it manually:
pip install pymupdf
```

## Running the Application

1. **Start Ollama** (if not already running):
   ```bash
   # On Linux/macOS
   ollama serve
   
   # On Windows
   # Ollama typically runs as a service after installation
   ```

2. **Launch the Streamlit app**:
   ```bash
   # Make sure your virtual environment is activated
   # The command prompt should show (venv) at the beginning
   
   # Start the app
   streamlit run app.py
   ```

3. The application will open in your default web browser, typically at `http://localhost:8501`

## Features

- Upload and analyze multiple images and PDF documents using Gemma 3 vision model
- Choose between general description and custom field extraction
- Process PDF files page by page or as complete documents
- Extract specific fields like invoice numbers, dates, and amounts
- View detailed descriptions of each uploaded file
- Download analysis results as a CSV file for further processing

## Technical Details

This application leverages several key technologies:

- **Streamlit**: Provides the web interface and interactive components
- **Ollama**: Runs the Gemma 3 vision model locally 
- **Gemma 3 12B**: Google's vision-language model that powers the image analysis
- **Pillow (PIL)**: Handles image processing and format conversion
- **PyMuPDF**: Converts PDF pages to images for processing (imported as `fitz`)
- **Pandas**: Used for data manipulation and CSV export

## Troubleshooting

- **Package installation issues**: If you encounter problems installing the dependencies, try updating pip first: `pip install --upgrade pip`
- **Compatibility issues**: This application works best with Python 3.9-3.12. Python 3.13 might have compatibility issues with some packages.
- **Ollama connection errors**: Ensure Ollama is running by checking `ollama serve` or restarting the Ollama service.
- **PDF processing errors**: If PDF processing isn't working, ensure PyMuPDF is installed correctly with `pip install pymupdf`.
- **PDF support optional**: The application will run without PDF support, but will display a warning. Install PyMuPDF to enable PDF processing.

## Try to keep your prompts short with Gemma for better results, larger promps with detaild instructions don't seem to result in better outputs
---

Made with ❤️ by Adrian with Claude - [ad1x.com](https://ad1x.com)
