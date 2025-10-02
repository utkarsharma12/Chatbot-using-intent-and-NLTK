# Chatbot-using-intent-and-NLTK
# 🤖 AI Chatbot with Intents-based Training

This project is a simple **AI-powered chatbot** built using **Python, NLTK, and TensorFlow/Keras**.  
The chatbot is trained on an **intents JSON file**, where each intent contains example user inputs (*patterns*) and bot replies (*responses*).

---

## 🚀 Features
- Intent-based classification of user queries  
- Preprocessing with **NLTK (tokenization, lemmatization)**  
- Training using a **neural network** (Keras + TensorFlow)  
- Responses stored in a simple **`intents.json`** file  
- Ability to **expand training data dynamically**  
- Lightweight and easy to extend  

---

## 📂 Project Structure
chatbot/
│── intents.json # Training data (intents, patterns, responses)
│── chatbot.py # Main chatbot script
│── train.py # Script to train the model
│── model.pkl # Trained model (saved)
│── words.pkl # Vocabulary
│── classes.pkl # Intent classes
│── README.md # Documentation


---

## 🛠️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/chatbot.git
   cd chatbot
2. Install dependencies:

pip install numpy tensorflow nltk

3. Download NLTK data (run this in Python shell once):

import nltk
nltk.download('punkt')
nltk.download('wordnet')


📖 Usage

1. Train the chatbot (only needed if you modify intents.json):

python train.py


2. Run the chatbot:

python chatbot.py


Example:

You: Hello
Bot: Hi there! How can I help you?

3. 🧠 Expanding Knowledge

Add new patterns & responses in intents.json.

Retrain the model with:

python train.py


You can also enable learning while chatting by allowing the bot to save unknown inputs back to intents.json.

4. 🔮 Future Improvements

Use word embeddings (Word2Vec, GloVe, or BERT)

Connect with APIs (e.g., Weather, News, Database)

Deploy on web apps or messaging apps (Telegram, WhatsApp, etc.)

Implement real-time continuous learning

📜 License

This project is open-source and available under the MIT License.

👨‍💻 Developed by: Utkarsh

