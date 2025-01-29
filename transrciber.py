# Install necessary packages
# !pip install openai-whisper
# !pip install torch
# !pip install soundfile
# !pip install numpy
# !pip install librosa

import whisper
import soundfile as sf
import numpy as np
import torch

# Load the Whisper model
model_size = "base"  # You can change this to "tiny", "small", "medium", or "large"
model = whisper.load_model(model_size)

def transcribe_audio(file_path):
    # Load the audio file
    audio, sample_rate = sf.read(file_path, dtype=np.float32)
    
    # Convert stereo to mono if necessary
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Resample to 16kHz if necessary (Whisper expects 16kHz audio)
    if sample_rate != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    
    # Transcribe the audio
    result = model.transcribe(audio)
    return result["text"]

# Path to your audio file
audio_file_path = "common_voice_id_41274039.mp3"

# Transcribe the audio
transcription = transcribe_audio(audio_file_path)
print("Transcription:", transcription)

# Save the transcription to a text file
output_file_path = "transcription.txt"
with open(output_file_path, "w") as f:
    f.write(transcription)

print(f"Transcription saved to {output_file_path}")