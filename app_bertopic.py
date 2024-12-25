import os
import logging
from flask import Flask, request, jsonify
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bertopic import BERTopic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Enable logging
logging.basicConfig(level=logging.INFO)

# Initialize NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Flask application
app = Flask(__name__)

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, text):
        tokens = nltk.word_tokenize(text)
        tokens = [word.lower() for word in tokens]
        tokens = [word for word in tokens if word not in self.stop_words]
        tokens = [self.stemmer.stem(word) for word in tokens]
        return ' '.join(tokens)

class JobClusterModel:
    def __init__(self):
        self.model = BERTopic()
        self.top_keywords = {}
        self.topic_labels = {}  # Dictionary to map topic IDs to labels
        self.department_mapping = {}  # Dictionary to map topic IDs to departments

    def fit(self, job_descriptions):
        self.topics, _ = self.model.fit_transform(job_descriptions)
        self._calculate_keywords_and_define_topics()
        self.top_keywords = self.model.get_topic_info()

    def predict(self, job_description):
        topics, _ = self.model.transform([job_description])
        return topics[0]

    def get_keywords(self, topic):
        topic_info = self.model.get_topic(topic)
        if topic_info:
            return [word for word, _ in topic_info]
        return []

    def _calculate_keywords_and_define_topics(self):
        topic_info = self.model.get_topic_info()
        for topic in range(len(topic_info)):
            keywords = self.get_keywords(topic)
            self.topic_labels[topic] = ', '.join(keywords[:3])  # Use top keywords as the label for simplicity

        # Manually mapping topics to departments based on keywords inspection
        self.department_mapping = {
            0: "Engineering",
            1: "Support",
            2: "Data Science",
            3: "Marketing",
            # We can add more departments here, as of now I am mapping only 4 of them
        }

    def get_department(self, topic):
        return self.department_mapping.get(topic, "Unknown")

def load_dataset(file_path):
    logging.info("Loading dataset from %s", file_path)
    df = pd.read_csv(file_path)
    return df['Job Description'].dropna()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        logging.info("Received request: %s", data)
        
        job_description = data.get('job_description')
        if not job_description:
            logging.error("Job description not provided")
            return jsonify({'error': 'Job description not provided'}), 400

        preprocessed_text = preprocessor.preprocess(job_description)
        topic = model.predict(preprocessed_text)
        department = model.get_department(topic)

        logging.info("Predicted department: %s", department)
        response = {
            'department': department,
            'keywords': model.get_keywords(topic)
        }
        return jsonify(response)

    except Exception as e:
        logging.exception("Error during prediction")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    dataset_path = os.getenv("DATASET_PATH", "./data/Booking_Jobs_All_220218.csv")
    port = int(os.getenv("FLASK_PORT", 6000))

    job_descriptions = load_dataset(dataset_path)
    preprocessor = TextPreprocessor()
    processed_descriptions = job_descriptions.apply(preprocessor.preprocess)

    model = JobClusterModel()
    model.fit(processed_descriptions)

    app.run(debug=True, port=port)
