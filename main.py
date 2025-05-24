import time

from transformers import pipeline

# Load the pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model="./whisper-small-kyrgyz-local",
    # device=0  # use device=0 for GPU, or remove this line for CPU
)

# Transcribe an audio file
result = pipe("mix.wav")

# Print the transcription
start = time.time()
print("start")
print("Transcription:", result["text"])
print(time.time() - start)
