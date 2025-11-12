# 🎮 Fortnite Match Streamer Finder

A utility script designed to scan the player list in the **Fortnite Report/Spectator window** and identify which players are currently broadcasting their gameplay on Twitch.

[![GitHub license](https://img.shields.io/github/license/YOUR_USERNAME/YOUR_REPO_NAME?color=green)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📝 Functionality Overview

This tool automates the process of checking potential streamers in your match. It performs the following steps:

1.  **Screen Capture & Scroll:** Captures a designated area (the player list) and simulates smooth mouse wheel scrolling to view all entries.
2.  **OCR (Name Extraction):** Uses **Tesseract OCR** and advanced image preprocessing to extract player usernames from the captured images.
3.  **Twitch API Query:** Contacts the Twitch Helix API using your credentials to check the live status of the extracted names.
4.  **Result Display:** Presents a list of unique players found. If a player is streaming, a button is provided to open their Twitch channel directly.

### Robust Scrolling Mechanism

The script uses a **multi-threaded architecture (Producer/Consumer model)** to keep the screen capture and scrolling process highly fluid, even while the slow tasks (OCR and network calls) run in the background.

Crucially, it uses **Binarized Mean Squared Error (MSE)** on a cropped area of the player list to reliably detect when the list has reached its end, preventing infinite scrolling regardless of dynamic backgrounds in the game window.

---

## 🌟 Demo in Action

See the script in action—check out the fluidity of the scroll and the **precise, immediate stop** when the end of the list is detected:

<img src="https://github.com/user-attachments/assets/39e29f97-911c-40c7-abf9-f31abc77b595" width="500">

---

## 🛠️ Installation and Setup Guide

To use this script, you require Python libraries, the Tesseract OCR utility, and Twitch API keys.

### Step 1: Python and Dependencies

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
    cd YOUR_REPO_NAME
    ```
2.  **Install Python Libraries:**
    All required Python packages are listed in `requirements.txt`. Install them using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Installing and Configuring Tesseract OCR (Essential)

Tesseract is an **external program** required for text recognition. It cannot be installed via `pip`.

1.  Download and install [**Tesseract OCR**](https://github.com/UB-Mannheim/tesseract/wiki).
2.  **Locate the `tesseract.exe` executable.** (e.g., `C:\Program Files\Tesseract-OCR\tesseract.exe`).
3.  **Configure the Path:** You do **not** need to add Tesseract to your system's environment PATH. Instead, you must specify the exact location of the executable within the Python script itself.
4.  **Edit `fortnite_stream_finder.py`:** Open the file and modify the `TESSERACT_CMD` variable in the configuration section to your local path:

    ```python
    # fortnite_stream_finder.py (in the CONFIG section)
    TESSERACT_CMD = r"C:\YOUR\LOCAL\INSTALL\PATH\tesseract.exe" 
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    ```

### Step 3: Twitch API Credentials

To perform live status checks, you need to provide your Twitch application keys.

1.  Create an application in your [Twitch Developer Console](https://dev.twitch.tv/console/apps).
2.  Obtain your **Client ID** and **Client Secret**.
3.  Enter these keys into the designated input fields within the application's graphical interface when you run the script.

## 🚀 Usage Instructions

1.  **Launch the Script:**
    ```bash
    python fortnite_stream_finder.py
    ```
2.  **In Fortnite:** Open the **Report/Spectator window** that displays the list of players.
3.  **Select Region:** Click the **"⤓ Select region"** button in the script's GUI and accurately drag a box around the **player list**. Ensure the cursor stays within the list's boundary.
4.  **Start Scanning:** Enter your Twitch credentials (Step 3) and click **"▶ Start scan"**.
    * The cursor will automatically move to the center of the selected area to ensure the game window has scroll focus.
    * The script will begin capturing, scrolling, and processing the player names in the background.
5.  **View Results:** Names will appear in the results panel. Any player currently live will show a clickable button to navigate to their Twitch channel.
6.  **Stop:** The scan stops automatically when the bottom of the list is reached. You can manually stop at any time by pressing **F4** or the **"■ Stop"** button.

---
