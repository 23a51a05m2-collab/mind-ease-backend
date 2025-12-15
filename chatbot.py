import json
import random
import re
import string
import joblib
import numpy as np

# === NEW: This is our text cleaning function ===
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

class ChatBot:
    def __init__(self, model_path, intents_path):
        try:
            # Load the trained model and its components
            self.model_data = joblib.load(model_path)
            self.model = self.model_data['model'] # This is the full pipeline
            # self.vectorizer = self.model_data['vectorizer'] # <- THIS LINE IS REMOVED
            
            with open(intents_path, 'r') as f:
                self.intents = json.load(f)

        except FileNotFoundError:
            print(f"Error: Model file '{model_path}' or intents file '{intents_path}' not found.")
            print("Please make sure you have run 'python train_model.py' first.")
            exit()
        except KeyError:
            # === NEW: Add a specific error for the old model file ===
            print(f"Error: Model file '{model_path}' is in an old format.")
            print("Please re-run 'python train_model.py' to create an updated model.")
            exit()
        except Exception as e:
            print(f"Error loading model: {e}")
            exit()

    def get_response(self, message):
        """
        Predicts the intent of the message and returns a response.
        """
        # === NEW: We define a confidence threshold ===
        CONFIDENCE_THRESHOLD = 0.5 # 50%
        
        # Clean the user's message using the *same* function
        cleaned_message = clean_text(message)
        
        # Handle empty string after cleaning (e.g., user just typed "??")
        if not cleaned_message.strip():
             return random.choice(self.get_fallback_response())

        try:
            # === THIS IS THE FIX ===
            # We pass the *raw text* to the pipeline's predict_proba
            # The pipeline ('self.model') handles vectorizing internally
            predictions = self.model.predict_proba([cleaned_message])[0]
            
            # message_vec = self.vectorizer.transform([cleaned_message]) # <- THIS LINE IS REMOVED
            # predictions = self.model.predict_proba(message_vec)[0] # <- THIS WAS THE BUG

            # Get the highest prediction
            best_match_index = np.argmax(predictions)
            confidence = predictions[best_match_index]
            intent_tag = self.model.classes_[best_match_index]

            # === NEW: Print debug info to the *terminal* ===
            print(f"[ChatBot Log] Message: '{message}' -> Cleaned: '{cleaned_message}' -> Intent: '{intent_tag}' (Confidence: {confidence:.4f})")

            # === NEW: Check if confidence is high enough ===
            if confidence > CONFIDENCE_THRESHOLD:
                # We are confident, find a response for that intent
                for intent in self.intents['intents']:
                    if intent['tag'] == intent_tag:
                        return random.choice(intent['responses'])
            
            # If confidence is too low, use the default fallback
            return random.choice(self.get_fallback_response())

        except Exception as e:
            print(f"Error during prediction: {e}")
            return "I'm having a little trouble understanding right now."

    def get_fallback_response(self):
        """
        Finds and returns a list of fallback responses.
        """
        for intent in self.intents['intents']:
            if intent['tag'] == 'default_fallback':
                return intent['responses']
        # Absolute fallback if 'default_fallback' tag is missing
        return ["I'm not sure I understand. Can you rephrase?"]