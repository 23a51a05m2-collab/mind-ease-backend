import json
import re
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# === NEW: This is the *exact same* cleaning function from chatbot.py ===
def clean_text(text):
    """
    Cleans text by:
    1. Lowercasing
    2. Removing punctuation
    3. Removing extra whitespace
    """
    text = text.lower() # 1. Lowercase
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text) # 2. Remove punctuation
    text = " ".join(text.split()) # 3. Remove extra whitespace
    return text

def train_model():
    print("Starting model training...")
    
    try:
        with open('intents.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: 'intents.json' not found. Please make sure it's in the same folder.")
        return

    X_train = []
    y_train = []

    # === NEW: We clean the patterns before training ===
    for intent in data['intents']:
        for pattern in intent['patterns']:
            # Apply the cleaning function to every pattern
            cleaned_pattern = clean_text(pattern)
            
            # Avoid adding empty strings if a pattern was just "!" or "?"
            if cleaned_pattern:
                X_train.append(cleaned_pattern)
                y_train.append(intent['tag'])

    if not X_train:
        print("Error: No valid training data found after cleaning. Check 'intents.json'.")
        return

    print(f"Training on {len(X_train)} cleaned patterns.")

    # Create a pipeline
    # We use TfidfVectorizer for text and LogisticRegression as the classifier
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
        ('classifier', LogisticRegression(random_state=42, C=5, max_iter=500, solver='lbfgs'))
    ])

    # --- Optional but recommended: Hyperparameter tuning ---
    # We can search for the best 'C' value for Logistic Regression
    param_grid = {
        'classifier__C': [1, 5, 10]
    }
    
    # Use GridSearchCV to find the best parameters
    # cv=3 means 3-fold cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
    
    print("Running GridSearchCV... This might take a moment.")
    grid_search.fit(X_train, y_train)

    # Get the best model from the search
    best_model = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    # Save all the necessary components
    model_data = {
        'model': best_model
        # We only need to save the 'model' (which is the full pipeline)
        # 'vectorizer': best_model.named_steps['vectorizer'] # <- THIS LINE IS REMOVED
    }
    
    joblib.dump(model_data, 'model.joblib')
    
    print("\nTraining complete! Model saved as 'model.joblib'.")

if __name__ == "__main__":
    train_model()