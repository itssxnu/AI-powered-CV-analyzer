# Setup Guide: CV Parser & Skill Matcher

This guide walks you through setting up the CV Model backend in a new PyCharm environment on a Windows machine.

## Prerequisites

Before starting, ensure you have **Python 3.9 or higher** installed and added to your system `PATH`.
You can verify this by running `python --version` in your terminal.

---

## Part 1: Installing Required System Dependencies

This project relies on `pytesseract` and `pdf2image`, which require external system dependencies to be installed first.

### 1. Install Tesseract OCR
1. Download the latest Windows installer from UB Mannheim: [Tesseract OCR for Windows](https://github.com/UB-Mannheim/tesseract/wiki).
2. Run the installer and proceed with the default settings.
   - Pay attention to the installation folder (usually `C:\Program Files\Tesseract-OCR`).
3. **Add Tesseract to your PATH:**
   - Press the Windows Key and search for `Environment Variables`.
   - Click **Edit the system environment variables**.
   - Click the **Environment Variables...** button.
   - Under *System variables* (or *User variables*), find the variable named **Path**, select it, and click **Edit**.
   - Click **New** and paste the installation path (e.g., `C:\Program Files\Tesseract-OCR`).
   - Click **OK** to save and close all windows.

### 2. Install Poppler
1. Download the latest Poppler Windows binaries from here: [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/).
2. Extract the downloaded zip file.
3. Move the extracted folder to a permanent location (e.g., `C:\Program Files\poppler`).
4. **Add Poppler to your PATH:**
   - Go to **Environment Variables** -> **Path** -> **Edit** (same as above).
   - Click **New** and paste the path to the `bin` folder inside your Poppler directory (e.g., `C:\Program Files\poppler\Library\bin` or similar, depending on extraction).
   - Click **OK** to save and close all windows.

*Note: You may need to restart your computer or command prompts for the new PATH variables to take effect.*

---

## Part 2: PyCharm Project Setup

1. **Extract your project:** Make sure your project `.zip` file is extracted into a folder on your new PC.
2. **Open in PyCharm:** Open PyCharm. Click **Open** from the start screen, and select the extracted project folder.
3. **Run the Setup Script:**
   - In PyCharm, open the **Terminal** tab at the bottom.
   - Run the provided exact setup script by typing: 
     ```bash
     .\setup_env.bat
     ```
   - This script will automatically create a virtual environment (`.venv`), upgrade `pip`, and install all the project dependencies listed in `requirements.txt`.

### Linking the Virtual Environment in PyCharm
If PyCharm does not automatically detect your new virtual environment:

1. Go to **File > Settings** (or press `Ctrl + Alt + S`).
2. Navigate to **Project: [Your Project Name] > Python Interpreter** on the left menu.
3. Click **Add Interpreter** (or the gear icon ⚙️) -> **Add Local Interpreter**.
4. Choose **Existing environment**.
5. Click the `...` button to browse, and navigate to your project folder. Select the Python executable located at:
   `[Your Project Folder]\.venv\Scripts\python.exe`
6. Click **OK** to confirm. 

PyCharm will now index your project and resolve all Python dependencies automatically!

---

## Part 3: Environment Variables (.env)
Don't forget to create a `.env` file in the root of your project directory based on any API keys or specific configuration your project needs to connect properly.
