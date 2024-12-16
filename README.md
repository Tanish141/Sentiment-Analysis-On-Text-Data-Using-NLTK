### 🎬 Sentiment Analysis on Movie Reviews
Welcome to the Sentiment Analysis on Movie Reviews project! This project uses a Naive Bayes Classifier to classify movie reviews as positive or negative based on their sentiment. 📊

---

### 🚀 Features
🔄 Preprocessing: Tokenization and removal of stopwords from text data.
📚 Movie Reviews Dataset: Leveraging the labeled NLTK movie reviews corpus for training and testing.
🛠️ Naive Bayes Classifier: A simple and efficient model for text classification.
📈 Model Evaluation: Achieves an accuracy of ~75% on the test set and highlights the most informative features.
✨ Custom Input Testing: Analyze the sentiment of any input text in real-time!

---

### 🛠️ Technologies Used
Python 🐍
NLTK (Natural Language Toolkit) 📚

---

### 📂 Dataset
Source: NLTK's movie_reviews corpus, containing 2,000 labeled movie reviews.
Preprocessing Steps:
Tokenization using nltk.word_tokenize.
Stopword removal with nltk.corpus.stopwords.
Feature extraction by converting words into a dictionary format.

---

### ⚙️ How It Works
🗂 Load Dataset: The movie reviews dataset is tokenized and preprocessed.
🤖 Train Model: A Naive Bayes Classifier is trained on labeled reviews.
🔍 Evaluate Model: The classifier is tested on a separate dataset, with metrics like accuracy reported.
💬 Custom Testing: Input any sentence to classify its sentiment as positive or negative.

---

### 📜 Results
🎯 Accuracy: Approximately 75% on the test set.
🏆 Most Informative Features: Highlights words with the highest predictive power.

---

### 🚀 How to Run
Clone the repository:
bash
Copy code
git clone https://github.com/Tanish141/Sentiment-Analysis-Movie-Reviews.git  
Navigate to the project directory:
bash
Copy code
cd Sentiment-Analysis-Movie-Reviews  
Install dependencies:
bash
Copy code
pip install -r requirements.txt  
Download required NLTK datasets:
python
Copy code
import nltk  
nltk.download('movie_reviews')  
nltk.download('punkt')  
nltk.download('stopwords')  
Run the script:
bash
Copy code
python SentimentAnalysisOnTextData.py  

---

### 🎨 Visualization
Most Informative Features: Displays the top 10 words contributing to sentiment classification.
Predicted Sentiments: Outputs sentiment predictions for custom inputs in an easy-to-read format.

---

### 🏅 Key Learning Outcomes
Preprocessing text data for sentiment analysis.
Training and testing a Naive Bayes Classifier.
Tokenizing and removing stopwords with NLTK.
Analyzing model accuracy and informative features.

---

### 🤝 Contributions
Feel free to open issues or submit pull requests to enhance this project. Let's collaborate and make it better! ✨

---

### 📧 Contact
If you have any questions or suggestions, reach out at:
Email: mrtanish14@gmail.com
GitHub: Tanish141

---

### 🎉 Happy Coding!
