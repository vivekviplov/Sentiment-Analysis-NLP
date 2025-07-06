# Sentiment-Analysis-NLP

This project is a sentiment analysis model built using Natural Language Processing (NLP) techniques on the IMDB movie reviews dataset. It involves preprocessing text data through tokenization, stopword removal, and lemmatization, followed by feature extraction using TF-IDF Vectorization. The processed data is then fed into a Support Vector Machine (SVM) classifier, and hyperparameters are optimized using GridSearchCV for better performance. The final model is evaluated using metrics such as the F1-score and confusion matrix. After tuning and testing, the model achieved a final F1-score of 0.89, demonstrating strong performance in classifying reviews as positive or negative.
📂 Dataset
Source: IMDB Movie Review Dataset

Size: 50,000 labeled reviews

Balance: Equal distribution of positive and negative sentiments

🔧 Technologies & Libraries
Python (Pandas, NumPy)

Scikit-learn (SVC, LinearSVC, GridSearchCV)

NLTK (for stopwords and lemmatization)

Matplotlib & Seaborn (for visualization)

TF-IDF Vectorization

⚙️ Process Overview
Data Preprocessing

Removed punctuation

Lowercased text

Removed stopwords

Lemmatized tokens

Vectorization

Used TfidfVectorizer to convert text to feature vectors

Modeling

Trained models using SVC with different kernels (linear, poly, sigmoid)

Performed hyperparameter tuning with GridSearchCV

Evaluation

Accuracy Score

Confusion Matrix

F1 Score

🧠 Key Findings
Linear kernel was fast and efficient on high-dimensional TF-IDF data

RBF and Sigmoid kernels gave better accuracy but were computationally slower

Best Accuracy: (Add your best score here)
