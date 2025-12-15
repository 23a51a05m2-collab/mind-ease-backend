from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import ChatBot # We import the class

print("Starting Flask server...")

app = Flask(__name__)
origins = [
    "https://fanciful-pony-461d2a.netlify.app/" 
]
CORS(app, resources={r"/chat": {"origins": origins}})

# --- THIS IS THE FIX ---
# We now pass the file paths to the ChatBot constructor
try:
    bot = ChatBot(model_path='model.joblib', intents_path='intents.json')
    print("ChatBot loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load ChatBot. {e}")
    # We exit if the bot can't load, as the server is useless.
    exit()
# ----------------------

@app.route("/")
def home():
    return "Hello! Your MindEase backend server is running."

@app.route("/chat", methods=['POST'])
def chat():
    try:
        data = request.json
        
        if not data or 'message' not in data:
            return jsonify({"error": "No message provided."}), 400
            
        message = data['message']
        
        # Get response from the bot
        response = bot.get_response(message)
        
        return jsonify({"response": response})

    except Exception as e:
        print(f"[Server Error] {e}")
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == '__main__':
    # We set debug=False for a cleaner terminal, 
    # as our chatbot now provides its own logs.

    app.run(host='127.0.0.1', port=5000, debug=False)
