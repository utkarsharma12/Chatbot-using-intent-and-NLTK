import random
import json
import pickle
import numpy as np
import os
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import datetime
import requests
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv() 

try:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in .env file")
    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("[INFO] Gemini model loaded successfully.")

except Exception as e:
    print(f"[ERROR] Could not configure Gemini. Please check your API key and .env file. Error: {e}")
    gemini_model = None


def download_nltk_resources():
    """Downloads necessary NLTK resources if they are not found."""
    resources = ["punkt", "wordnet", "omw-1.4"]
    for resource in resources:
        try:
            if resource == "punkt":
                nltk.data.find(f"tokenizers/{resource}")
            else:
                nltk.data.find(f"corpora/{resource}")
        except LookupError:
            print(f"[INFO] Downloading NLTK resource: {resource} ...")
            nltk.download(resource, quiet=True)

download_nltk_resources()

lemmatizer = WordNetLemmatizer()

with open("intents.json") as file:
    intents = json.load(file)

if not os.path.exists("models"):
    os.makedirs("models")

words_file = "models/words.pkl"
classes_file = "models/classes.pkl"
model_file = "models/chatbot_model.h5"

def train_model():
    words = []
    classes = []
    documents = []
    ignore_letters = ["?", "!", ".", ","]
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent["tag"]))
            if intent["tag"] not in classes:
                classes.append(intent["tag"])
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))
    pickle.dump(words, open(words_file, "wb"))
    pickle.dump(classes, open(classes_file, "wb"))
    training = []
    output_empty = [0] * len(classes)
    for doc in documents:
        bag = []
        word_patterns = doc[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for w in words:
            bag.append(1 if w in word_patterns else 0)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])
    random.shuffle(training)
    training = np.array(training, dtype=object)
    train_x = np.array(list(training[:, 0]))
    train_y = np.array(list(training[:, 1]))
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation="softmax"))
    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    print("[INFO] Training model, please wait...")
    hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
    model.save(model_file, hist)
    print("[INFO] Model trained and saved!")

if not os.path.exists(model_file):
    train_model()

model = load_model(model_file)
words = pickle.load(open(words_file, "rb"))
classes = pickle.load(open(classes_file, "rb"))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow_vector = bow(sentence, words)
    res = model.predict(np.array([bow_vector]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json, user_message):
    tag = intents_list[0]["intent"] if intents_list else None
    
    if tag == "general_query" or not tag:
        if gemini_model:
            try:
                response = gemini_model.generate_content(user_message)
                return response.text
            except Exception as e:
                print(f"Error calling Gemini API: {e}")
                return "I'm having a little trouble connecting to my super-brain right now. Please try again."
        else:
            return "My connection to the Gemini model is not set up. Please check the API key."
    
    if tag == "time":
        now = datetime.datetime.now()
        time_str = now.strftime("%I:%M %p")
        return f"The current time is {time_str}."
        
    if tag == "date":
        today = datetime.date.today()
        date_str = today.strftime("%B %d, %Y")
        return f"Today's date is {date_str}."

    if tag == "weather":
        api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        if not api_key:
            return "OpenWeatherMap API key not configured."
        city = "Greater Noida"
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
        complete_url = f"{base_url}q={city}&appid={api_key}&units=metric"
        try:
            response = requests.get(complete_url)
            weather_data = response.json()
            if weather_data["cod"] != "404":
                main_data = weather_data["main"]
                current_temperature = main_data["temp"]
                weather_description = weather_data["weather"][0]["description"]
                return f"The weather in {city} is currently {int(current_temperature)}°C with {weather_description}."
            else:
                return "Sorry, I couldn't find the weather for that city."
        except Exception:
            return "Sorry, I'm having trouble connecting to the weather service."

    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            return random.choice(i["responses"])

def chatbot_response(msg):
    ints = predict_class(msg)
    res = get_response(ints, intents, msg)
    return res

if __name__ == "__main__":
    print("Chatbot is ready! (type 'quit' to exit)")
    while True:
        message = input("You: ")
        if message.lower() == "quit":
            print("Chatbot: Goodbye!")
            break
        response = chatbot_response(message)
        print("Chatbot:", response)
