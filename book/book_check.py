import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder

# Φόρτωσε τα δεδομένα
df = pd.read_csv('Book_Details.csv')

# Μετατροπή genres
df['genres_list'] = df['genres'].apply(ast.literal_eval)

# Ορισμός target variable - "δημοφιλές" vs "όχι δημοφιλές"
median_threshold = df['num_ratings'].median()
df['is_popular'] = (df['num_ratings'] > median_threshold).astype(int)

print(f"Target variable created: {df['is_popular'].sum()} popular books out of {len(df)}")

print("🔧 FEATURE ENGINEERING")
print("=" * 40)

def create_features(df):
    features = pd.DataFrame(index=df.index)
    
    features['average_rating'] = df['average_rating']
    
    author_stats = df.groupby('author').agg({
        'num_ratings': ['mean', 'count'],
        'average_rating': 'mean'
    }).reset_index()
    
    author_stats.columns = ['author', 'author_avg_ratings', 'author_book_count', 'author_avg_rating']
    df = df.merge(author_stats, on='author', how='left')
    
    features['author_avg_ratings'] = df['author_avg_ratings']
    features['author_book_count'] = df['author_book_count']
    features['author_avg_rating'] = df['author_avg_rating']
    
    features['num_genres'] = df['genres_list'].apply(len)
    
    def extract_pages(x):
        try:
            if isinstance(x, list) and len(x) > 0:
                return int(x[0].split()[0])
            return 0
        except:
            return 0
    
    features['pages'] = df['num_pages'].apply(extract_pages)
    
    def extract_year(x):
        try:
            if isinstance(x, list) and len(x) > 0:
                year_str = x[0].split()[-1]
                return int(year_str) if year_str.isdigit() else 0
            return 0
        except:
            return 0
    
    features['publication_year'] = df['publication_info'].apply(extract_year)
    
    return features.fillna(0)

X = create_features(df)
y = df['is_popular']

print(f"📊 Features shape: {X.shape}")
print(f"🎯 Target shape: {y.shape}")
print(f"📋 Features columns: {list(X.columns)}")

print(f"🔍 Missing values: {X.isnull().sum().sum()}")

print("✅ FEATURE ENGINEERING COMPLETE!")

# Training ML Model
print("🤖 TRAINING ML MODEL")
print("=" * 40)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Training set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples")

# ML Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"📈 Accuracy: {accuracy:.3f}")

print("✅ MODEL TRAINING COMPLETE!")

# Model Analysis & Feature Importance
print("📊 MODEL ANALYSIS")
print("=" * 40)

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("🔍 FEATURE IMPORTANCE:")
print(feature_importance)

# Detailed Evaluation
from sklearn.metrics import confusion_matrix, classification_report

print("\n📈 DETAILED EVALUATION:")
print(classification_report(y_test, y_pred, target_names=['Not Popular', 'Popular']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n🎯 CONFUSION MATRIX:")
print(f"True Negatives: {cm[0,0]} | False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]} | True Positives: {cm[1,1]}")

print("✅ MODEL ANALYSIS COMPLETE!")

# Predictions & Real-world Testing
print("🔮PREDICTIONS & TESTING")
print("=" * 40)

def predict_popularity(book_title, model, df, X):
    if book_title not in df['book_title'].values:
        return "Book not found"
    
    book_idx = df[df['book_title'] == book_title].index[0]
    features = X.iloc[book_idx:book_idx+1]
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    result = {
        'book': book_title,
        'predicted_popular': prediction,
        'confidence': max(probability),
        'actual_ratings': df.loc[book_idx, 'num_ratings'],
        'actual_popular': df.loc[book_idx, 'is_popular']
    }
    
    return result

test_books = [
    "Harry Potter and the Half-Blood Prince",
    "To Kill a Mockingbird", 
    "The Great Gatsby",
    "Changeling"  # Λιγότερο δημοφιλό
]

print("🧪 PREDICTION TESTS:")
for book in test_books:
    result = predict_popularity(book, model, df, X)
    if isinstance(result, dict):
        status = "✅ Correct" if result['predicted_popular'] == result['actual_popular'] else "❌ Wrong"
        print(f"{status} | {result['book']}:")
        print(f"   Prediction: {'Popular' if result['predicted_popular'] else 'Not Popular'} ({result['confidence']:.1%})")
        print(f"   Actual: {result['actual_ratings']} ratings ({'Popular' if result['actual_popular'] else 'Not Popular'})")
        print()

print("✅ PREDICTION TESTING COMPLETE!")