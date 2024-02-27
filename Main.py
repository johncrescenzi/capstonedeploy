import joblib
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, jsonify, render_template, request, send_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from wordcloud import WordCloud

app = Flask(__name__)

# Load the CSV data
data = pd.read_csv('mbti.csv')

# Preprocess the data (you need to implement this part based on your specific preprocessing requirements)

# Split the data into features (comments) and labels (MBTI types)
X = data['posts']
y = data['type']

# Define a pipeline for feature extraction and model training
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC()),
])

# Train the model
pipeline.fit(X, y)

# Load the pre-trained NLP model
model = joblib.load('mbti_nlp_model.pkl')


def evaluate_model(user_comment, X_train, X_test, y_train, y_test):
  # Fit the TfidfVectorizer with the new training data
  pipeline.named_steps['tfidf'].fit(X_train)

  # Train the model
  pipeline.fit(X_train, y_train)

  # Predict using the input comment
  y_pred = pipeline.predict([user_comment])

  # Calculate accuracy
  accuracy = accuracy_score(y_test, y_pred)
  return accuracy


# Define a function to predict MBTI type based on user input
def predict_mbti(user_comment):
  prediction = model.predict([user_comment])
  return prediction[0]


# Define a function to generate a wordcloud
def generate_wordcloud(text):
  # Generate the word cloud
  wordcloud = WordCloud(width=1600, height=800,
                        background_color='white').generate(text)
  return wordcloud


# Define the route for the home page
@app.route('/')
def index():
  return render_template('index.html')


# Route for predicting MBTI type and sending accuracy based on user input
@app.route('/predict', methods=['POST'])
def predict():
  user_comment = request.form['userComment']

  # Predict MBTI type
  predicted_mbti = predict_mbti(user_comment)

  return jsonify({'predicted_mbti': predicted_mbti})


# Visualization routes
# Route for sending the distribution of MBTI types visualization
@app.route('/distribution')
def send_distribution():
  # Generate the visualization
  plt.figure(figsize=(10, 6))
  data['type'].value_counts().plot(kind='bar')
  plt.title('Distribution of MBTI Types')
  plt.xlabel('MBTI Type')
  plt.ylabel('Count')
  plt.xticks(rotation=45)

  # Save the visualization to a temporary file
  filename = 'distribution.png'
  plt.savefig(filename)
  plt.close()

  # Send the visualization file to the frontend
  return send_file(filename, mimetype='image/png')


# Route for sending the word clouds visualization
@app.route('/wordclouds')
def send_wordclouds():
  # Generate the visualization
  plt.figure(figsize=(15, 10))
  for i, mbti_type in enumerate(data['type'].unique(), 1):
    plt.subplot(4, 4, i)
    wordcloud = generate_wordcloud(' '.join(
        data[data['type'] == mbti_type]['posts']))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(mbti_type)
    plt.axis("off")

  # Save the visualization to a temporary file
  filename = 'wordclouds.png'
  plt.savefig(filename)
  plt.close()

  # Send the visualization file to the frontend
  return send_file(filename, mimetype='image/png')


# Route for sending the post length visualization
@app.route('/postlength')
def send_postlength():
  # Generate the visualization
  plt.figure(figsize=(10, 6))
  data['post_length'] = data['posts'].apply(lambda x: len(x.split()))
  data.groupby('type')['post_length'].mean().plot(kind='bar')
  plt.title('Average Post Length by MBTI Type')
  plt.xlabel('MBTI Type')
  plt.ylabel('Average Post Length')
  plt.xticks(rotation=45)

  # Save the visualization to a temporary file
  filename = 'postlength.png'
  plt.savefig(filename)
  plt.close()

  # Send the visualization file to the frontend
  return send_file(filename, mimetype='image/png')


if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8080)
