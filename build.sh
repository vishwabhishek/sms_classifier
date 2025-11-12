#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "
import nltk
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
"

echo "Build completed successfully!"
