import whisper
import sys
import sounddevice as sd
import numpy as np
import wave

def record_audio(filename, duration=5, samplerate=44100):
    print("Recording...")
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    print("Recording finished.")
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())

def transcribe_audio(file_path):
    # Load the Whisper model
    model = whisper.load_model("large-v3")  # You can use 'base', 'small', 'medium', or 'large-v3'
    
    # Transcribe the audio file
    result = model.transcribe(file_path)
    
    # Print the transcribed text
    print("Transcription:")
    print(result["text"])
    
    return result["text"]

def main():
    audio_file = "recorded_audio.wav"
    record_audio(audio_file)
    transcribe_audio(audio_file)

if __name__ == "__main__":
    main()
