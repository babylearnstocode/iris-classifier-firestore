# Iris Classifier with Firestore Integration

This project is an interactive web application for classifying iris flowers using a machine learning model, with data storage and retrieval powered by Google Firestore.

## Features

- Predicts iris species from user input features
- Stores predictions and user data in Firestore
- Displays prediction history
- Built with Streamlit for easy web deployment

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd iris_classfier_firestore_integration
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   (Make sure you have Python 3.7+ and pip installed)

3. **Set up Firestore credentials:**
   - Place your Firebase service account key JSON file in a secure location.
   - Set the environment variable in your shell or `.env` file:
     ```bash
     export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/serviceAccountKey.json"
     ```

## Running the App

Start the Streamlit app with:

```bash
streamlit run app/streamlit_app.py
```

Then open the provided local URL in your browser to interact with the app.

## Project Structure

```
app/
  streamlit_app.py
  ...
requirements.txt
README.md
```

## License

MIT License
