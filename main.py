import streamlit as st
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
import scipy.io.wavfile
import numpy as np


# Load the processor and model
@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    return processor, model


processor, model = load_model()


def generate_music(prompt, length):
    # Prepare inputs
    inputs = processor(text=[prompt], padding=True, return_tensors="pt")

    # Generate audio tokens
    with torch.no_grad():
        generated_tokens = model.generate(**inputs, max_length=length)

    # Assuming generated_tokens are in the expected format
    audio_output = generated_tokens.squeeze().cpu().numpy()

    return audio_output, model.config.audio_encoder.sampling_rate


def save_audio(audio_data, sampling_rate, filename="generated_music.wav"):
    # Save the generated music as a .wav file
    scipy.io.wavfile.write(filename, rate=sampling_rate, data=audio_data)


# Streamlit App Layout
st.title("AI Music Generator 🎶")
st.write("Generate music based on your input description!")

# Sidebar for music length adjustment
st.sidebar.header("Settings")
music_length = st.sidebar.slider("Select music length (max tokens)", min_value=128, max_value=512, value=256, step=64)

# User prompt input
user_input = st.text_input("Enter the description of the music you want:")

if st.button("Generate Music"):
    if user_input:
        with st.spinner("Generating music..."):
            try:
                # Generate music
                audio_data, sampling_rate = generate_music(user_input, music_length)

                # Convert audio to WAV format (for playback and download)
                wav_audio = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)  # Normalize audio to int16

                # Display audio player in Streamlit
                st.audio(wav_audio, sample_rate=sampling_rate)

                # Download option
                st.download_button(
                    label="Download Music",
                    data=wav_audio.tobytes(),
                    file_name="generated_music.wav",
                    mime="audio/wav"
                )
            except Exception as e:
                st.error(f"Error generating music: {str(e)}")
    else:
        st.warning("Please enter a music description before generating!")

# Footer (optional visibility)
st.markdown("""
    <style>
    footer {visibility: hidden;}  /* Remove this line if you want the footer visible */
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

#! pip install streamlit -q
#!wget -q -O - ipv4.icanhazip.com
#! streamlit run app.py & npx localtunnel --port 8501

#
#