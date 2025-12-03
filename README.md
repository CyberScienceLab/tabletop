# How to Run #


1. Run the application:

python app.py

2. Open the following URL in your browser:

http://127.0.0.1:7860/


## How to Run on Windows (CMD)

Follow the commands below to download the repository, install dependencies, and run the application:

```cmd
git clone https://github.com/CyberScienceLab/tabletop.git
cd tabletop
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
python app.py
```

Before running the application, set your API key in the .env file, for example:
OPENAI_API_KEY=your_api_key_here
