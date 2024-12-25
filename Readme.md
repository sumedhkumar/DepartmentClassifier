# Department Classifier

## Overview
This project aims to create an unsupervised model that classifies job listings into their respective departments based solely on the "Job Description" column. This approach can be particularly useful for organizing unstructured job data into coherent categories.

## Dataset
The dataset used for this project can be found on Kaggle:
[Booking.com Jobs EDA & NLP Ensemble Modeling](https://www.kaggle.com/code/niekvanderzwaag/booking-com-jobs-eda-nlp-ensemble-modeling/data)

## Tasks and Questions

### 1. Text Pre-Processing
Perform text pre-processing steps on the "Job Description" column and explain the utility of each step in the context of this task.

### 2. Cluster Identification
Identify the number of natural clusters present in the data.

### 3. Model Training
Train an unsupervised model to classify the jobs into their respective departments using only the "Job Description" column.

### 4. Keyword Identification
Identify key words from each cluster that are indicative of the department.

### 5. Model Deployment
Deploy the trained model in two different ways:
a. As a REST API endpoint

## Running the Application

### Prerequisites
- Python 3.7 or higher
- Required Python packages (listed in `requirements.txt`)

### Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/department-classifier.git
    cd department-classifier
    ```

2. Set up a virtual environment and activate it (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate    # On Windows, use "venv\Scripts\activate"
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Configuration
1. Create a `.env` file in the root directory of the project. This file should contain any environment variables required for the application. Here is an example of what the `.env` file might look like:
    ```txt
    FLASK_APP=app.py
    FLASK_ENV=development
    ```

2. Ensure the `.env` file is correctly configured with all necessary values. Required env variables given in .env_example file. 

### Running the Flask API
1. To start the REST API server, run:
    ```sh
    flask run   # Or simply use python app_kmeans.py
    ```

The application should now be running at `http://127.0.0.1:6000/`.

## Input/Output Payloads

### Input Payload
The input payload should be in the following format:
```json
{
    "job_description": "Your job description"
}
```
### Expected Output
The expected output will be in the following format:
```json
{
    "department": "Engineering",
    "keywords": [
        "work",
        "experi",
        "product",
        "world",
        "travel",
        "develop",
        "manag",
        "team",
        "opportun",
        "data"
    ]
}
```

### Explanation of Fields
- job_description: The job description text which needs to be classified.
- department: The department to which the job description is classified.
- keywords: A list of key words/phrases that are indicative of the department. These keywords are extracted from the job description and reflect common terms associated with the identified department.

## Tasks Performed
The following tasks were performed in this repository:

1. Text Pre-Processing: Steps such as tokenization, stop word removal, stemming, and lemmatization were performed on the "Job Description" column to prepare the text data for clustering.

2. Cluster Identification: Various clustering methods were explored to identify natural clusters in the data.

3. Model Training: Three different clustering methods were tried - KMeans, DBSCAN, and BERTopic. Among these, BERTopic performed significantly better compared to the others in terms of creating coherent and meaningful clusters.

4. Keyword Identification: Key words from each cluster were identified to indicate the department. These keywords help in understanding and labeling the clusters effectively.

5. Model Deployment: The trained model was deployed as a REST API endpoint and guidelines were provided to deploy.



# Contact
For any questions or further assistance, please raise an issue on the GitHub repository or contact the maintainer at [sumedh.bhalerao07@gmail.com].
