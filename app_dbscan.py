import os
from flask import Flask, request, jsonify
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import logging
from dotenv import load_dotenv

nltk.download('punkt')
nltk.download('stopwords')

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Text Preprocessing Class
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

# Load and preprocess the data
def load_dataset(file_path):
    logging.info("Loading dataset from %s", file_path)
    df = pd.read_csv(file_path)
    job_descriptions = df['Job Description'].dropna()
    preprocessor = TextPreprocessor()
    processed_descriptions = job_descriptions.apply(preprocessor.preprocess)
    return processed_descriptions

def get_top_keywords(vectorizer, model, num_keywords=10):
    feature_names = vectorizer.get_feature_names_out()
    top_keywords = {}
    for cluster_label in set(model.labels_):
        if cluster_label == -1:
            continue
        indices = [i for i, label in enumerate(model.labels_) if label == cluster_label]
        centroid = model.components_[cluster_label]
        words = [feature_names[idx] for idx in centroid.argsort()[-num_keywords:]]
        top_keywords[cluster_label] = words
    return top_keywords

# Load data and train the model
def initialize_model(dataset_path, max_features=1000, epsilon=0.5, min_samples=5):
    job_descriptions = load_dataset(dataset_path)
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(job_descriptions).toarray()
    
    model = DBSCAN(eps=epsilon, min_samples=min_samples, metric='cosine')
    model.fit(X)
    
    keywords = get_top_keywords(vectorizer, model)
    return vectorizer, model, keywords

# Map cluster labels to departments
def map_clusters_to_departments(top_keywords):
    cluster_department_mapping = {}
    for cluster, keywords in top_keywords.items():
        if any(word in keywords for word in ["engineer", "developer", "software"]):
            cluster_department_mapping[cluster] = "Engineering"
        elif any(word in keywords for word in ["support", "help", "service"]):
            cluster_department_mapping[cluster] = "Support"
        elif any(word in keywords for word in ["data", "science", "analysis"]):
            cluster_department_mapping[cluster] = "Data Science"
        elif any(word in keywords for word in ["marketing", "brand", "campaign"]):
            cluster_department_mapping[cluster] = "Marketing"
        else:
            cluster_department_mapping[cluster] = "Miscellaneous"
    return cluster_department_mapping

# Load dataset path and parameters from environment variables
dataset_path = os.getenv("DATASET_PATH", "./data/Booking_Jobs_All_220218.csv")
tfidf_max_features = int(os.getenv("TFIDF_MAX_FEATURES", 1000))
dbscan_eps = float(os.getenv("DBSCAN_EPSILON", 0.5))
dbscan_min_samples = int(os.getenv("DBSCAN_MIN_SAMPLES", 5))

# Initialize the model
vectorizer, model, top_keywords = initialize_model(
    dataset_path,
    max_features=tfidf_max_features,
    epsilon=dbscan_eps,
    min_samples=dbscan_min_samples
)

# Create cluster to department mapping
cluster_department_mapping = map_clusters_to_departments(top_keywords)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        logging.info("Received prediction request with data: %s", data)

        job_description = data.get('job_description')
        if not job_description:
            logging.error("Job description not provided in the request data")
            return jsonify({'error': 'Job description not provided'}), 400

        preprocessor = TextPreprocessor()
        preprocessed_text = preprocessor.preprocess(job_description)
        X_new = vectorizer.transform([preprocessed_text]).toarray()
        cluster = int(model.fit_predict(X_new)[0])  # Convert to int

        if cluster == -1:
            logging.warning("Job description classified as noise")
            return jsonify({'cluster': cluster, 'error': 'This data point is considered noise'}), 400

        department = cluster_department_mapping.get(cluster, "Unknown")
        logging.info("Cluster: %s, Department: %s, Keywords: %s", cluster, department, top_keywords.get(cluster, []))
        
        response = {
            'cluster': cluster,
            'department': department,
            'keywords': top_keywords.get(cluster, [])
        }
        return jsonify(response)

    except Exception as e:
        logging.exception("Exception occurred during prediction")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("FLASK_PORT", 6000))
    app.run(debug=True, port=port)
