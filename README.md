# Advanced Music Analysis Tool

This project is an advanced music analysis tool that leverages AI to provide detailed insights into audio files. It offers comprehensive analysis of musical structure, technical aspects, sonic elements, and sampling potential, making it valuable for musicians, producers, and audio enthusiasts.

## Features

- Detailed analysis of audio structure and composition
- Identification of key musical elements and sections
- Tempo, key, and time signature detection
- Sampling recommendations with creative use suggestions
- Audio splitting based on analysis results

## Installation

1. Clone this repository to your local machine.

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Open `main.py` and replace `'YOUR_API_KEY'` with your actual Gemini API key:
   ```python
   GEMINI_API_KEY = 'YOUR_API_KEY'
   ```

## Usage

1. Start the FastAPI server:
   ```
   python main.py
   ```

2. Open the `home.html` file located in the `frontend` folder in your web browser.

3. Use the web interface to upload audio files and receive detailed analysis results.

## API Endpoints

- `/analyze-audio`: POST request to analyze an uploaded audio file
- `/split-samples`: POST request to split audio based on sampling recommendations

## Supported Audio Formats

- MP3
- WAV
- OGG

## Note

This tool uses AI-powered analysis and may require significant processing time for longer audio files. Ensure you have a stable internet connection for API communication.
