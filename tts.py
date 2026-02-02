import os
from config import OPENAI_API_KEY
import simpleaudio as sa

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from pathlib import Path
from openai import OpenAI

client = OpenAI()
speech_file_path = str(Path(__file__).parent / "speech.wav")
script = "You're doing a really good job maintaining a straight left arm throughout your swing, which is great! Your posture is solid, and your stance gives you a good base. Just a little tip—try to keep your head steadier during the swing. It moves quite a bit, and staying more still will help you hit the ball more cleanly. Keep up the good work, and you'll improve even faster!"
# script = 'hi baby'
with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="ash",
    input=script,
    instructions="Speak like an encouraging golf instructor, offering clear and constructive feedback with a confident and supportive tone.",
    response_format="wav",
) as response:
    response.stream_to_file(speech_file_path)

# sa.WaveObject.from_wave_file(speech_file_path).play().wait_done()
