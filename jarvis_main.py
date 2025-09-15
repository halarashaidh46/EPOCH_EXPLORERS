#jarvis_main.py
import gradio as gr
import requests
from datetime import datetime
import wikipedia
import numpy as np
import speech_recognition as sr
from transformers import pipeline
from scipy.io.wavfile import write

# ---------------- Text-to-Speech Initialization ---------------- #
tts = pipeline("text-to-speech", model="facebook/mms-tts-eng")

from scipy.io.wavfile import write
import tempfile

def text_to_speech(answer):
    try:
        speech = tts(answer)
        audio_array = None
        if "audio" in speech:
            audio_array = np.array(speech["audio"], dtype=np.float32)
        elif "array" in speech:
            audio_array = np.array(speech["array"], dtype=np.float32)
        if audio_array is None:
            return None

        # Convert float32 [-1,1] to int16 and save to temporary WAV file
        audio_int16 = np.clip(audio_array, -1, 1) * 32767
        audio_int16 = audio_int16.astype(np.int16)
        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        write(tmp_wav.name, speech["sampling_rate"], audio_int16)
        return tmp_wav.name  # <-- return path for Gradio
    except Exception as e:
        print("TTS error:", e)
        return None


# ---------------- Helper Functions ---------------- #

def get_weather(city):
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        geo_resp = requests.get(geo_url).json()
        if "results" not in geo_resp:
            return f"Sorry, I couldn't find the city '{city}'."
        lat = geo_resp["results"][0]["latitude"]
        lon = geo_resp["results"][0]["longitude"]
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_resp = requests.get(weather_url).json()
        if "current_weather" not in weather_resp:
            return "Sorry, I couldn't fetch the weather right now."
        temp = weather_resp["current_weather"]["temperature"]
        return f"The weather in {city.title()} is {temp}°C."
    except Exception as e:
        print("Weather error:", e)
        return "Error fetching weather."

from datetime import datetime
from zoneinfo import ZoneInfo  # Python 3.9+

def get_time():
    jordan_time = datetime.now(ZoneInfo("Asia/Amman"))
    return f"The current time is {jordan_time.strftime('%H:%M:%S')} and today's date is {jordan_time.strftime('%A, %B %d, %Y')}."

def search_wikipedia(query):
    try:
        return wikipedia.summary(query, sentences=2, auto_suggest=True)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Topic too broad. Options: {e.options[:5]}"
    except wikipedia.exceptions.PageError:
        return "Sorry, I couldn't find anything."
    except Exception as e:
        print("Wikipedia error:", e)
        return "Error fetching Wikipedia."

def detect_intent(text):
    text_lower = text.lower()
    if "weather" in text_lower:
        return "weather"
    elif "time" in text_lower or "date" in text_lower:
        return "time"
    elif "summarize" in text_lower or "summary" in text_lower:
        return "summarize"
    elif "search" in text_lower:
        return "search"
    else:
        return "chat"

# ---------------- Main Jarvis Function ---------------- #

def jarvis_voice(audio_file):
    try:
        print("=== jarvis_voice called ===")
        print("audio_file (filepath):", audio_file, type(audio_file))

        if audio_file is None:
            print("No audio received.")
            return "No audio received. Please speak something.", None

        # --- Speech-to-Text ---
        r = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)
        try:
            query = r.recognize_google(audio)
            print("STT successful:", query)
        except Exception as e:
            print("STT error:", e)
            query = ""

        if not query:
            print("STT produced empty text.")
            return "Sorry, I couldn't understand you.", None

        # --- Intent Detection & Response ---
        intent = detect_intent(query)
        print("Detected intent:", intent)

        if intent == "weather":
            city = query.lower().split("in")[-1].strip() if "in" in query.lower() else "Amman"
            answer = get_weather(city)
        elif intent == "time":
            answer = get_time()
        elif intent == "summarize":
            answer = "Please paste a text to summarize (not supported yet)."
        elif intent == "search":
            query_ans = query.lower().split("search", 1)[-1].strip() if "search" in query.lower() else ""
            answer = search_wikipedia(query_ans) if query_ans else "What do you want me to search?"
        else:
            answer = "I can tell you the time, weather, or search Wikipedia."

        print("Answer generated:", answer)

        # --- Convert Answer to Speech ---
        try:
            tts_wav_path = text_to_speech(answer)
            print("TTS WAV path:", tts_wav_path)
        except Exception as e:
            print("TTS conversion failed:", e)
            tts_wav_path = None

        if tts_wav_path is None:
            print("No TTS audio generated.")
            return answer, None

        print("Returning answer and audio path to Gradio.")
        return answer, tts_wav_path

    except Exception as e:
        print("Exception caught in Jarvis:", e)
        return f"Error inside Jarvis: {str(e)}", None

# ---------------- Gradio Interface ---------------- #

import gradio as gr

usage_text = """
### Welcome to the Face Recognition Assistant! 🤖
You can interact with it using voice commands:
1. Say **'search <something>'**  → to search online  
2. Say **'date'**               → to get today's date  
3. Say **'time'**               → to get current time  
4. Say **'weather in <city>'**  → to get weather info for a city  
**Example:** 'search Python tutorials' or 'weather in Amman'
"""

with gr.Blocks() as iface:
    # Instructions at the top
    gr.Markdown(usage_text)

    # Inject CSS for the button
    gr.HTML("""
    <style>
        #hey_btn {
            background-color: #020339 !important;
            color: white !important;
            font-weight: bold;
            border-radius: 12px !important;
        }
    </style>
    """)

    with gr.Row():
        with gr.Column():
            audio_in = gr.Audio(label="🎤 Speak", type="filepath")
            btn = gr.Button("Hey Jarvis!", elem_id="hey_btn")  # custom button

        with gr.Column():
            text_out = gr.Textbox(label="Jarvis says", lines=6)
            audio_out = gr.Audio(label="🔊 Audio Output")

    btn.click(fn=jarvis_voice, inputs=audio_in, outputs=[text_out, audio_out])

if __name__ == "__main__":
    iface.launch()