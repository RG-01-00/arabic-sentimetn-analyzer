# ARABIC SENTIMENT ANALYZER

What if you could give an Arabic sentence to an AI and let it understand whether the meaning is positive, negative, neutral, or mixed?

This project is an Arabic sentiment analysis platform that uses machine learning to understand the emotional meaning of Arabic text.

For example, you can give it a sentence such as:

هذا التطبيق رائع وسهل الاستخدام

The system analyzes the text and returns information about its sentiment, confidence, important keywords, and other linguistic insights.

The machine learning model was trained on approximately 20,000 Arabic text examples. The project combines a fine-tuned Arabic BERT model, TF-IDF analysis, and additional NLP techniques to produce more detailed results.

## THE SYSTEM CAN DETECT:

* Positive sentiment
* Negative sentiment
* Neutral sentiment
* Mixed sentiment
* Negation and its effect on sentiment
* Potential sarcasm
* Important keywords using TF-IDF
* Sentiment confidence and probability scores
* Arabic and mixed Arabic-English text

## PROJECT OVERVIEW

Arabic is a complex language to analyze automatically. Different writing styles, dialects, expressions, and context can change the meaning of a sentence.

This project combines machine learning and NLP techniques to build a system capable of analyzing Arabic text and returning meaningful sentiment insights.

The main sentiment model is a fine-tuned Arabic BERT model trained on approximately 20,000 examples.

TF-IDF is also used to identify important keywords in the analyzed text.

## HOW IT WORKS:

### RUNNING THE BACKEND

1. Open a terminal and enter the backend folder:

```bash
cd backend
```

2. Create and activate a virtual environment if needed.

On Windows:

```bash
.venv\Scripts\activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Start the backend:

```bash
uvicorn main:app --reload
```

The backend will be available at:

```text
http://127.0.0.1:8000
```

You can access the interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

You can test the machine learning model directly from the API documentation without running the frontend.

### RUNNING THE FRONTEND

Open another terminal and enter the frontend folder:

```bash
cd frontend
```

Install the dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The frontend will normally be available at:

```text
http://localhost:5173
```

## API EXAMPLE

### Endpoint:

```text
POST /analyze
```

### Request:

```json
{
  "text": "هذا التطبيق رائع وسهل الاستخدام"
}
```

The API returns information such as the detected sentiment, confidence score, sentiment probabilities, important keywords, and other analysis results.

## TRAINING DATA

The original training dataset and training scripts are not included in this repository.

The trained model is included and is already prepared to run with the backend. The model does not need to be trained again before using the application.

## TECHNOLOGIES

### Backend

* Python
* FastAPI
* PyTorch
* Hugging Face Transformers
* Arabic BERT
* Scikit-learn
* TF-IDF

### Frontend

* React
* Vite
* JavaScript
* XLSX
* Lucide React

## TESTING THE BACKEND WITHOUT THE FRONTEND

The frontend is not required to test the machine learning system.

Run the backend and open:

```text
http://127.0.0.1:8000/docs
```

From the interactive API documentation, you can send Arabic text directly to the `/analyze` endpoint and view the returned results.

The backend can therefore be used independently as an Arabic sentiment analysis API.
