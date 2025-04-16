import os
from config import OPENAI_API_KEY
import simpleaudio as sa

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from pathlib import Path
from openai import OpenAI

client = OpenAI()
speech_file_path = str(Path(__file__).parent / "speech.wav")
script = "Remember to keep your left arm straight and rotate your hips through the swing. Great job staying balanced!"
# script = 'hi baby'
with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="ash",
    input=script,
    instructions="Speak like an encouraging golf instructor, offering clear and constructive feedback with a confident and supportive tone.",
    response_format="wav",
) as response:
    response.stream_to_file(speech_file_path)

sa.WaveObject.from_wave_file(speech_file_path).play().wait_done()
