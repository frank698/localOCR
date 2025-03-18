## Installation and setup

### Linux
**Setup Ollama**:
   ```bash
   # setup ollama on linux 
   curl -fsSL https://ollama.com/install.sh | sh
   # pull gemma-3 vision model
   ollama run gemma3:12b
   ```

### macOS
**Setup Ollama**:
   ```bash
   # Install Ollama using Homebrew
   brew install ollama
   # Or download from https://ollama.com/download
   
   # pull gemma-3 vision model
   ollama run gemma3:12b
   ```

### Windows
**Setup Ollama**:
   1. Download the Ollama installer from [https://ollama.com/download](https://ollama.com/download)
   2. Run the installer and follow the on-screen instructions
   3. Open Command Prompt or PowerShell and run:
   ```powershell
   # pull gemma-3 vision model
   ollama run gemma3:12b
   ```

**Install Dependencies** (All platforms):
   Ensure you have Python 3.11 or later installed.
   ```bash
   # On Linux/macOS
   pip3 install streamlit ollama pillow
   
   # On Windows
   pip install streamlit ollama pillow
   ```

---
