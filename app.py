import speech_recognition as sr  # For speech recognition using Google AI
import pyttsx3  # For voice output
import os
import numpy as np
from chatbot import chat_bot


class VoiceAssistant:
    def __init__(self):
        self.assistant_name = ""
        print("Welcome to  Voice Assistant 😊")

    def set_assistant_name(self, name):
        self.assistant_name = name

    def get_assistant_name(self):
        return self.assistant_name

    def listen(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening... 🤔")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio_data = recognizer.listen(source, timeout=15)
            text = "Error"
        try:
            text = recognizer.recognize_google(audio_data)
            print(f"User -> {text}")
            return text
        except sr.RequestError as e:
            print(f"Error -> Could not request results; {e} 😞")
            return text
        except sr.UnknownValueError:
            print("Error -> Unknown error occurred 😞")
            return text

    def speak(self, text):
        print(f" Alexa -> {text} ")
        engine = pyttsx3.init()
        voices = engine.getProperty("voices")
        engine.setProperty("voice", voices[1].id)
        engine.say(text)
        engine.runAndWait()

    def get_response(self, text):
        response = chat_bot(text)
        return response


if __name__ == "__main__":
    assistant = VoiceAssistant()

    while True:
        assistant.speak(
            "Welcome to  Voice Assistant! My name is Alexa. How can I assist you today? "
        )
        assistant.speak("Would you like to chat or speak with me? 🤔")
        mode = int(input("(1 for chat, 2 for voice interaction) \nUser -> "))

        if mode == 1:
            name = input("Alexa -> What would you like to call me? 😊\nUser -> ")
            assistant.set_assistant_name(name)
            while True:
                user_input = input("User -> ")
                if any(
                    word in user_input.lower()
                    for word in ["quit", "exit", "close", "shut down", "bye"]
                ):
                    break
                else:
                    response = assistant.get_response(user_input)
                    print(f" Alexa -> {response} ")

        elif mode == 2:
            assistant.speak("What would you like to call me? 😊")
            while True:
                response = assistant.listen()
                if response == "Error":
                    assistant.speak("Sorry, could you please repeat that? 😞")
                else:
                    break
            assistant.set_assistant_name(response)
            assistant.speak("Great! What's on your mind? 🤔")
            while True:
                user_input = assistant.listen()
                if any(word in user_input.lower() for word in ["thank", "thanks"]):
                    response = np.random.choice(
                        [
                            "You're welcome! 😄",
                            "Anytime! 😊",
                            "No problem! 😄",
                            "Cool! 😉",
                            "I'm here if you need me! 😊",
                            "Mention not! 😄",
                        ]
                    )
                    assistant.speak(response)
                elif any(
                    word in user_input.lower() for word in ["your name", "who are you"]
                ):
                    response = f"I'm {assistant.get_assistant_name()} 😊"
                    assistant.speak(response)
                elif any(
                    word in user_input.lower()
                    for word in ["exit", "close", "quit", "bye"]
                ):
                    break
                else:
                    if user_input == "Error":
                        assistant.speak("Sorry, could you please repeat that? 😞")
                    else:
                        response = assistant.get_response(user_input)
                        assistant.speak(response)

        farewell_message = np.random.choice(
            [
                "Tata 🤗",
                "Have a good day 🤗",
                "Bye 🤗",
                "Goodbye 🤗",
                "Hope to meet soon 🤗",
                "Peace out 🤗!",
            ]
        )
        assistant.speak(farewell_message)
        break
