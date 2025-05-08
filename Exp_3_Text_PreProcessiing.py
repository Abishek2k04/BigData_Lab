import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.simplefilter('ignore')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    processed_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and 	word.isalpha()
    ]
    return ' '.join(processed_tokens)

documents = [
    "The quick brown fox jumped over the lazy dog.",
    "I love programming in Python, especially for data analysis.",
    "Natural language processing is fascinating!"
]
preprocessed_docs = [preprocess_text(doc) for doc in documents]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)
tfidf_array = tfidf_matrix.toarray()

feature_names = vectorizer.get_feature_names_out()
print("TF-IDF Matrix:")
for i, doc in enumerate(tfidf_array):
    print(f"\nDocument {i + 1}:")
    for word, score in zip(feature_names, doc):
        if score > 0:  
            print(f"{word}: {score}")
