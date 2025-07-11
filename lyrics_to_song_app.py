
# lyrics_to_song_app.py

import streamlit as st
from transformers import pipeline
import torchaudio
import os
import tempfile

# Define pipelines
lyrics_to_melody = pipeline("text-to-audio", model="facebook/musicgen-small")
tts = pipeline("text-to-speech", model="facebook/tts_transformer-es-css10")

# UI
st.set_page_config(page_title="Lyrics to Song Generator 🎵")
st.title("🎼 Lyrics to Song Generator")
st.markdown("Type in your lyrics, and we'll convert them into a song with background music!")

# Input
lyrics = st.text_area("Enter your lyrics here:", height=300)
voice_options = ["male", "female"]
voice_choice = st.radio("Choose a voice for singing:", voice_options)

# Generate song
if st.button("Generate Song 🎤") and lyrics:
    with st.spinner("Generating music and vocals..."):
        # Generate instrumental track
        music_output = lyrics_to_melody(lyrics, forward_params={"do_sample": True})

        # Save the instrumental
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_music:
            torchaudio.save(temp_music.name, music_output["audio"].unsqueeze(0), 32000)
            music_file = temp_music.name

        # Generate vocals (TTS)
        tts_output = tts(lyrics, forward_params={"voice": voice_choice})
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_vocal:
            torchaudio.save(temp_vocal.name, tts_output["audio"].unsqueeze(0), 22050)
            vocal_file = temp_vocal.name

        # Combine vocals + music using ffmpeg (external tool)
        final_song = os.path.join(tempfile.gettempdir(), "final_song.wav")
        os.system(f"ffmpeg -i {music_file} -i {vocal_file} -filter_complex amix=inputs=2:duration=longest {final_song} -y")

        # Show results
        st.audio(final_song, format="audio/wav")
        st.success("Done! Your song is ready 🎶")

        # Cleanup
        os.remove(music_file)
        os.remove(vocal_file)
        os.remove(final_song)

else:
    st.info("Enter some lyrics and click the button to generate a song.")
