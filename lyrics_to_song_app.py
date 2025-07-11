
import streamlit as st
from transformers import pipeline
import torch
import torchaudio
import tempfile
import os

# Initialize pipelines
st.title("🎼 Lyrics to Song Generator")
st.markdown("Type your lyrics and choose a voice. We’ll turn your lyrics into a song!")

with st.spinner("Loading models..."):
    melody_gen = pipeline("text-to-audio", model="facebook/musicgen-small")
    tts_gen = pipeline("text-to-speech", model="facebook/tts_transformer-es-css10")

lyrics = st.text_area("Enter your lyrics here", height=300)
voice = st.radio("Choose a voice:", ["male", "female"])

if st.button("Generate Song 🎤") and lyrics:
    with st.spinner("Generating background music..."):
        melody = melody_gen(lyrics, forward_params={"do_sample": True})
        melody_audio = melody["audio"].unsqueeze(0)  # Shape: [1, samples]
        melody_sr = 32000

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as music_file:
        torchaudio.save(music_file.name, melody_audio, sample_rate=melody_sr)
        st.audio(music_file.name, format="audio/wav", start_time=0)
        st.success("Background music generated!")

    with st.spinner("Generating vocal track..."):
        tts_out = tts_gen(lyrics, forward_params={"voice": voice})
        vocal_audio = tts_out["audio"].unsqueeze(0)
        vocal_sr = 22050

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as vocal_file:
        torchaudio.save(vocal_file.name, vocal_audio, sample_rate=vocal_sr)
        st.audio(vocal_file.name, format="audio/wav", start_time=0)
        st.success("Vocal audio generated!")

    st.markdown("""⚠️ Due to Streamlit Sharing limitations, music and vocal tracks are not mixed here.
Please download and combine them offline using Audacity or any audio editor.""")

