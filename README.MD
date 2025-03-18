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

- Upload and analyze multiple images using Gemma 3 vision model
- View detailed descriptions of each uploaded image
- Download analysis results as a CSV file for further processing

## Troubleshooting

- **Package installation issues**: If you encounter problems installing the dependencies, try updating pip first: `pip install --upgrade pip`
- **Compatibility issues**: This application works best with Python 3.9-3.12. Python 3.13 might have compatibility issues with some packages.
- **Ollama connection errors**: Ensure Ollama is running by checking `ollama serve` or restarting the Ollama service.

---
