from gtts import gTTS
import os
import platform
import subprocess

def speak_tamil(text):
    tts = gTTS(text=text, lang='ta')
    tts.save("tamil_speech.mp3")

    if platform.system() == "Windows":
        os.system("start tamil_speech.mp3")
    elif platform.system() == "Darwin":
        subprocess.call(["afplay", "tamil_speech.mp3"])
    else:
        subprocess.call(["xdg-open", "tamil_speech.mp3"])
