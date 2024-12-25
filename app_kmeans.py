import os
import logging
from flask import Flask, request, jsonify
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
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
    def __init__(self, n_clusters=4, max_features=1000):
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.model = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=42)
        self.top_keywords = {}
        self.cluster_labels = ["Engineering", "Support", "Data Science", "Marketing"]

    def fit(self, job_descriptions):
        X = self.vectorizer.fit_transform(job_descriptions).toarray()
        self.model.fit(X)
        self._extract_keywords()

    def predict(self, job_description):
        X_new = self.vectorizer.transform([job_description]).toarray()
        cluster = self.model.predict(X_new)[0]
        return cluster

    def _extract_keywords(self):
        cluster_centers = self.model.cluster_centers_
        feature_names = self.vectorizer.get_feature_names_out()
        for i in range(self.n_clusters):
            words = [feature_names[index] for index in cluster_centers[i].argsort()[-10:]]
            self.top_keywords[i] = words

    def get_keywords(self, cluster):
        return self.top_keywords.get(cluster, [])

    def get_cluster_label(self, cluster):
        return self.cluster_labels[cluster] if cluster < len(self.cluster_labels) else "Unknown"

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
        cluster = model.predict(preprocessed_text)
        cluster_label = model.get_cluster_label(cluster)

        logging.info("Predicted cluster: %d (%s)", cluster, cluster_label)
        return jsonify({
            'cluster': int(cluster),
            'cluster_label': cluster_label,
            'keywords': model.get_keywords(cluster)
        })

    except Exception as e:
        logging.exception("Error during prediction")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    dataset_path = os.getenv("DATASET_PATH", "./data/Booking_Jobs_All_220218.csv")
    n_clusters = int(os.getenv("N_CLUSTERS", 4))
    max_features = int(os.getenv("TFIDF_MAX_FEATURES", 1000))
    port = int(os.getenv("FLASK_PORT", 6000))

    job_descriptions = load_dataset(dataset_path)
    preprocessor = TextPreprocessor()
    processed_descriptions = job_descriptions.apply(preprocessor.preprocess)

    model = JobClusterModel(n_clusters=n_clusters, max_features=max_features)
    model.fit(processed_descriptions)

    app.run(debug=True, port=port)
