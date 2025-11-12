# Email/SMS Spam Classifier

A machine learning-powered web application that classifies emails and SMS messages as spam or legitimate using TF-IDF vectorization and a trained classification model.

## 🌟 Features

- **Real-time Classification**: Instantly classify any email or SMS message
- **Efficient Processing**: Optimized text preprocessing with caching
- **User-Friendly Interface**: Built with Streamlit for an intuitive web UI
- **NLP Pipeline**: Includes tokenization, stopword removal, and stemming
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## 🚀 Installation

### 1. Clone or Navigate to Project Directory

```bash
cd /home/abhishek-vishwakarma/Desktop/main_file
```

### 2. Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install streamlit nltk scikit-learn pandas
```

## 📦 Required Files

Make sure you have the following files in your project directory:

- `app.py` - Main Streamlit application
- `vectorizer.pkl` - Pre-trained TF-IDF vectorizer
- `model.pkl` - Pre-trained classification model

## 🏃 Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

The application will launch in your browser at:
- **Local URL**: http://localhost:8501
- **Network URL**: http://[YOUR_IP]:8501

## 📝 How to Use

1. **Enter Your Message**: Type or paste an email or SMS message in the text input field
2. **Get Classification**: The app will automatically process and classify the message
3. **View Result**: 
   - 🚨 **Spam** - Message is classified as spam
   - ✅ **Not Spam** - Message is legitimate

## 🔧 Project Structure

```
main_file/
├── app.py                 # Main application file
├── vectorizer.pkl         # TF-IDF vectorizer model
├── model.pkl             # Classification model
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## 🛠️ Technical Details

### Text Preprocessing Pipeline

The application performs comprehensive text preprocessing:

1. **Lowercasing**: Converts text to lowercase
2. **Tokenization**: Splits text into individual words
3. **Stopword Removal**: Removes common English words (the, a, an, etc.)
4. **Alphanumeric Filtering**: Keeps only alphanumeric characters
5. **Stemming**: Reduces words to their root form using Porter Stemmer
6. **Vectorization**: Converts preprocessed text to TF-IDF features

### Model Information

- **Vectorizer**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Classification Model**: Pre-trained classification model (likely Naive Bayes or Logistic Regression)
- **Output Classes**: 
  - 1 = Spam
  - 0 = Not Spam

## ⚙️ Performance Optimization

The application includes several optimization features:

- **Caching**: Models and stopwords are cached to avoid reloading
- **Resource Caching**: `@st.cache_resource` for ML models
- **Data Caching**: `@st.cache_data` for stopwords
- **Efficient Filtering**: Single-pass list comprehension for token filtering

## 🔐 NLTK Data

The application automatically downloads required NLTK data on first run:

- `punkt` tokenizer
- `stopwords` corpus

If automatic download fails, manually download:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 📊 Model Performance

The classification model achieves good accuracy on the training dataset. Performance depends on:

- Quality of training data
- Model type and hyperparameters
- Input text quality and format

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError

**Solution**: Install missing dependencies
```bash
pip install streamlit nltk scikit-learn pandas
```

### Issue: Model files not found

**Solution**: Ensure `vectorizer.pkl` and `model.pkl` are in the project directory

### Issue: NLTK data not downloading

**Solution**: Manually download using Python:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Issue: Port 8501 already in use

**Solution**: Specify a different port
```bash
streamlit run app.py --server.port 8502
```

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | Latest | Web UI framework |
| nltk | Latest | Natural Language Processing |
| scikit-learn | Latest | Machine Learning tools |
| pandas | Latest | Data handling |
| pickle | Built-in | Model serialization |

## 🔄 Workflow

```
User Input
    ↓
Text Preprocessing
    ↓
TF-IDF Vectorization
    ↓
Model Prediction
    ↓
Display Result (Spam/Not Spam)
```

## 💡 Tips for Better Results

1. **Clear Input**: Use clear, properly formatted text for best results
2. **Realistic Messages**: Test with real-world spam and legitimate messages
3. **Multiple Tests**: Try various message types to understand model behavior
4. **Meaningful Content**: Messages with very few words may not classify well

## 🤝 Contributing

Feel free to improve the model or UI:

- Retrain the model with more data
- Add confidence scores to predictions
- Implement batch processing
- Add message history/statistics

## 📄 License

This project is provided as-is for educational and practical use.

## 👤 Author

Created for Email/SMS Spam Classification

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Verify all dependencies are installed
3. Ensure model files exist in the directory

---

**Enjoy using your Spam Classifier!** 🎉
