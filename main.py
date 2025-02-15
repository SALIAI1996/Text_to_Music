import os
import streamlit as st
import torch
import scipy.io.wavfile
import numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# 🔹 Fix 1: Disable Streamlit's auto-reload to avoid PyTorch errors
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

# 🔹 Fix 2: Prevent PyTorch distributed memory issues (optional)
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

# 🔹 Fix 3: Use explicit attention implementation
@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained(
        "facebook/musicgen-small", 
        attn_implementation="eager"  # ✅ Fix for attention issue
    )
    return processor, model


processor, model = load_model()


def generate_music(prompt, length):
    # Prepare inputs
    inputs = processor(text=[prompt], padding=True, return_tensors="pt")

    # Generate audio tokens
    with torch.no_grad():
        generated_tokens = model.generate(**inputs, max_length=length)

    # Convert to numpy (dummy example since real conversion depends on model output format)
    audio_output = generated_tokens.squeeze().cpu().numpy()

    return audio_output, model.config.audio_encoder.sampling_rate


def save_audio(audio_data, sampling_rate, filename="generated_music.wav"):
    """ Save the generated music as a .wav file """
    scipy.io.wavfile.write(filename, rate=sampling_rate, data=audio_data)


# 🔹 Streamlit UI
st.title("🎵 AI Music Generator")
st.write("Generate music based on your input description!")

# 🔹 Sidebar for settings
st.sidebar.header("Settings")
music_length = st.sidebar.slider(
    "Select music length (max tokens)", min_value=128, max_value=512, value=256, step=64
)

# 🔹 User input prompt
user_input = st.text_input("Enter the description of the music you want:")

if st.button("Generate Music"):
    if user_input:
        with st.spinner("Generating music... 🎶"):
            try:
                # Generate music
                audio_data, sampling_rate = generate_music(user_input, music_length)

                # Convert audio to WAV format
                wav_audio = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)  # Normalize audio

                # Display audio player
                st.audio(wav_audio, sample_rate=sampling_rate)

                # Provide download option
                st.download_button(
                    label="⬇️ Download Music",
                    data=wav_audio.tobytes(),
                    file_name="generated_music.wav",
                    mime="audio/wav"
                )
            except Exception as e:
                st.error(f"⚠️ Error generating music: {str(e)}")
    else:
        st.warning("⚠️ Please enter a music description before generating!")

# 🔹 Hide Streamlit footer
st.markdown("""
    <style>
    footer {visibility: hidden;} 
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
