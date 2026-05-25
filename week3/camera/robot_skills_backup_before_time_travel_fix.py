import os
import json
import math
import socket
import urllib.parse
import urllib.request
from datetime import datetime

ROBOT_NAME = "Miguel"

def get_time_text():
    now = datetime.now()
    return f"It is {now.strftime('%I:%M %p').lstrip('0')}."

def get_date_text():
    now = datetime.now()
    return f"Today is {now.strftime('%A, %B %d, %Y')}."

def get_project_status_text():
    return (
        "Miguel's core systems are online. I can listen through the ReSpeaker, "
        "speak through the Creative speaker, see with the OAK-D Lite, detect faces, "
        "and use my cloud brain for smarter conversation."
    )

def check_internet_text():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return "Internet connection is online."
    except OSError:
        return "Internet connection appears to be offline."

def get_weather_text(location=None):
    location = location or os.getenv("MIGUEL_LOCATION", "Los Gatos, California")

    try:
        encoded = urllib.parse.quote(location)
        url = f"https://wttr.in/{encoded}?format=j1"

        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))

        current = data["current_condition"][0]
        temp_f = current.get("temp_F")
        feels_f = current.get("FeelsLikeF")
        desc = current["weatherDesc"][0]["value"]
        humidity = current.get("humidity")

        return (
            f"The weather in {location} is {temp_f} degrees Fahrenheit, "
            f"feels like {feels_f}, with {desc.lower()} and {humidity}% humidity."
        )

    except Exception as e:
        return f"I could not get the weather right now. Error: {e}"

def safe_calculate_text(text):
    # Very limited safe calculator: only numbers and math operators.
    expression = text.lower()
    replacements = {
        "plus": "+",
        "minus": "-",
        "times": "*",
        "multiplied by": "*",
        "divided by": "/",
        "over": "/",
        "x": "*",
    }

    for k, v in replacements.items():
        expression = expression.replace(k, v)

    allowed = set("0123456789.+-*/() ")
    cleaned = "".join(ch for ch in expression if ch in allowed).strip()

    if not cleaned:
        return "I could not find a calculation to solve."

    try:
        result = eval(cleaned, {"__builtins__": {}}, {"math": math})
        return f"The answer is {result}."
    except Exception:
        return "I could not solve that calculation yet."

def maybe_handle_local_skill(user_text, local_state=None):
    text = user_text.lower().strip()
    words = set(text.split())

    if not text:
        return None

    # Time skill: only answer clock-time questions.
    # Do NOT trigger for "time travel", "space time", "time machine", etc.
    time_phrases = [
        "what time is it",
        "what is the time",
        "current time",
        "time now",
        "what time now",
        "tell me the time",
    ]

    blocked_time_topics = [
        "time travel",
        "time traveling",
        "time machine",
        "space time",
        "spacetime",
        "time dilation",
    ]

    if any(blocked in text for blocked in blocked_time_topics):
        return None

    if any(phrase in text for phrase in time_phrases):
        return get_time_text()

    if "date" in words or "day" in words or "today" in words:
        return get_date_text()

    if "weather" in words or "temperature" in words:
        return get_weather_text()

    if "internet" in words or "wifi" in words or "network" in words:
        return check_internet_text()

    if "status" in words:
        vision = ""
        if local_state:
            if local_state.get("face_detected"):
                person = local_state.get("recognized_person")
                position = local_state.get("face_position")
                if person:
                    vision = f" I currently see {person} in the {position} position."
                else:
                    vision = f" I currently see a face in the {position} position."
            else:
                vision = " I do not currently see a face."
        return get_project_status_text() + vision

    if "calculate" in words or "how much" in text:
        if any(op in text for op in ["+", "-", "*", "/", "plus", "minus", "times", "divided"]):
            return safe_calculate_text(text)

    return None
