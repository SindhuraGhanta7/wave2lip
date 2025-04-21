#%%writefile app.py
import streamlit as st
import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
import time
from PIL import Image
import base64

# Set page configuration
st.set_page_config(
    page_title="Wav2Lip - AI Lip Sync Tool",
    page_icon="💋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4B4BFF;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    .success-text {
        color: #00CC00;
        font-weight: 500;
    }
    .stProgress > div > div > div > div {
        background-color: #FF4B4B;
    }
    .stDownloadButton button {
        background-color: #4B4BFF;
        color: white;
    }
    .parameter-section {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .results-section {
        background-color: #e6f7ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Function to get an HTML for the video player
def get_video_player_html(video_path):
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()
    video_b64 = base64.b64encode(video_bytes).decode()
    video_html = f"""
    <video width="100%" controls>
        <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    """
    return video_html

# Function to run the inference script
def run_inference(face_path, audio_path, checkpoint_path, output_path, static=False, fps=25,
                  pads=(0, 10, 0, 0), resize_factor=1, crop=(0, -1, 0, -1),
                  face_det_batch_size=16, wav2lip_batch_size=128, nosmooth=False, rotate=False):

    # Prepare command
    cmd = [
        "python", "inference.py",
        "--checkpoint_path", checkpoint_path,
        "--face", face_path,
        "--audio", audio_path,
        "--outfile", output_path,
        "--face_det_batch_size", str(face_det_batch_size),
        "--wav2lip_batch_size", str(wav2lip_batch_size),
        "--resize_factor", str(resize_factor),
    ]

    # Add optional parameters
    if static:
        cmd.append("--static")

    if fps != 25:
        cmd.extend(["--fps", str(fps)])

    if pads != (0, 10, 0, 0):
        cmd.extend(["--pads"] + [str(p) for p in pads])

    if crop != (0, -1, 0, -1):
        cmd.extend(["--crop"] + [str(c) for c in crop])

    if nosmooth:
        cmd.append("--nosmooth")

    if rotate:
        cmd.append("--rotate")

    # Run the command
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Display logs in real-time with a placeholder
    logs_placeholder = st.empty()

    # Process output in real-time
    full_log = []
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            full_log.append(output.strip())
            logs_placeholder.code('\n'.join(full_log))

    return_code = process.poll()

    # Check if the process was successful
    if return_code == 0:
        st.success(f"✅ Lip sync completed successfully! Output saved to {output_path}")
        return True, output_path
    else:
        error = process.stderr.read()
        st.error(f"❌ Error during lip sync process: {error}")
        return False, None

# Header
st.markdown("<h1 class='main-header'>🎬 Wav2Lip AI Lip Sync Tool</h1>", unsafe_allow_html=True)

# Description
st.markdown("""
This app uses Wav2Lip to sync lips in a video with a provided audio file.
Upload your video (or image) and audio, adjust parameters, and create a lip-synced video!
""")

# Create columns for input files
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h2 class='sub-header'>👤 Face Input</h2>", unsafe_allow_html=True)
    face_file = st.file_uploader("Upload video or image containing face", type=['mp4', 'jpg', 'png', 'jpeg'])
    if face_file:
        temp_face_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{face_file.name.split('.')[-1]}")
        temp_face_file.write(face_file.getvalue())
        temp_face_path = temp_face_file.name

        # Preview
        if face_file.name.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
            st.image(temp_face_path, caption="Face Image Preview", use_column_width=True)
        else:
            st.video(temp_face_path)

with col2:
    st.markdown("<h2 class='sub-header'>🔊 Audio Input</h2>", unsafe_allow_html=True)
    audio_file = st.file_uploader("Upload audio file", type=['wav', 'mp3', 'm4a', 'aac'])
    if audio_file:
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.name.split('.')[-1]}")
        temp_audio_file.write(audio_file.getvalue())
        temp_audio_path = temp_audio_file.name

        # Audio preview
        st.audio(temp_audio_path)

# Create a tab interface for advanced settings
st.markdown("<h2 class='sub-header'>⚙️ Configuration</h2>", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["Basic Settings", "Advanced Settings"])

with tab1:
    st.markdown("<div class='parameter-section'>", unsafe_allow_html=True)
    # Checkpoint selector
    st.subheader("Model Checkpoint")
    checkpoint_options = {
        "Wav2Lip-SD-GAN.pt": "/content/Wav2Lip/checkpoints/Wav2Lip-SD-GAN.pt",
        "Wav2Lip-SD-NOGAN.pt": "/content/Wav2Lip/checkpoints/Wav2Lip-SD-NOGAN.pt"
    }
    checkpoint_selection = st.selectbox(
        "Select checkpoint model",
        options=list(checkpoint_options.keys()),
        index=0,
        help="GAN version generally gives better quality but might be less stable"
    )
    checkpoint_path = checkpoint_options[checkpoint_selection]

    # Static image mode
    static_mode = st.checkbox(
        "Static Mode (use for single image input)",
        value=False,
        help="Use only the first frame for inference"
    )

    # FPS for static mode
    fps = st.number_input(
        "FPS (only applicable for static image input)",
        min_value=10,
        max_value=60,
        value=25,
        step=1,
        help="Frames per second for output video when using static image input"
    )

    # Resize factor
    resize_factor = st.slider(
        "Resize Factor (smaller numbers = higher resolution)",
        min_value=1,
        max_value=4,
        value=1,
        help="Reduce input resolution by this factor. Sometimes better results at lower resolutions."
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='parameter-section'>", unsafe_allow_html=True)
    # Face padding
    st.subheader("Face Padding")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pad_top = st.number_input("Top", value=0, min_value=0, max_value=100)
    with col2:
        pad_bottom = st.number_input("Bottom", value=10, min_value=0, max_value=100)
    with col3:
        pad_left = st.number_input("Left", value=0, min_value=0, max_value=100)
    with col4:
        pad_right = st.number_input("Right", value=0, min_value=0, max_value=100)

    # Batch sizes
    st.subheader("Batch Settings")
    col1, col2 = st.columns(2)
    with col1:
        face_det_batch_size = st.number_input(
            "Face Detection Batch Size",
            min_value=1,
            max_value=64,
            value=16,
            help="Batch size for face detection (reduce if you get OOM errors)"
        )
    with col2:
        wav2lip_batch_size = st.number_input(
            "Wav2Lip Batch Size",
            min_value=1,
            max_value=256,
            value=128,
            help="Batch size for Wav2Lip model (reduce if you get OOM errors)"
        )

    # Other options
    st.subheader("Additional Options")
    col1, col2 = st.columns(2)
    with col1:
        nosmooth = st.checkbox(
            "Disable Smoothing",
            value=False,
            help="Prevent smoothing face detections over time"
        )
    with col2:
        rotate = st.checkbox(
            "Rotate Video",
            value=False,
            help="Rotate video 90 degrees clockwise (for videos taken in portrait mode)"
        )
    st.markdown("</div>", unsafe_allow_html=True)

# Run button
st.markdown("### 🚀 Generate Lip-Synced Video")
if st.button("Start Processing", type="primary", use_container_width=True):
    if face_file and audio_file:
        with st.spinner("Processing... This may take a while depending on video length"):
            # Create output directory if it doesn't exist
            output_dir = "results"
            os.makedirs(output_dir, exist_ok=True)

            # Generate output filename
            output_filename = f"result_{int(time.time())}.mp4"
            output_path = os.path.join(output_dir, output_filename)

            # Run inference
            success, result_path = run_inference(
                face_path=temp_face_path,
                audio_path=temp_audio_path,
                checkpoint_path=checkpoint_path,
                output_path=output_path,
                static=static_mode,
                fps=fps,
                pads=(pad_top, pad_bottom, pad_left, pad_right),
                resize_factor=resize_factor,
                face_det_batch_size=face_det_batch_size,
                wav2lip_batch_size=wav2lip_batch_size,
                nosmooth=nosmooth,
                rotate=rotate
            )

            if success and result_path:
                st.markdown("<div class='results-section'>", unsafe_allow_html=True)
                st.markdown("## 🎉 Result")

                # Display the output video
                st.markdown(get_video_player_html(result_path), unsafe_allow_html=True)

                # Download button
                with open(result_path, "rb") as file:
                    st.download_button(
                        label="Download Result",
                        data=file,
                        file_name=output_filename,
                        mime="video/mp4",
                        use_container_width=True
                    )
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("Please upload both a face video/image and an audio file to continue.")

# Add footer with information
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p>
        Powered by Wav2Lip - An AI-based lip-sync model<br>
        <small>Upload limits depend on your system. For long videos, consider lowering the resolution with resize factor.</small>
    </p>
</div>
""", unsafe_allow_html=True)

# Clean up temporary files when the app stops
def cleanup():
    try:
        if 'temp_face_path' in locals():
            os.unlink(temp_face_path)
        if 'temp_audio_path' in locals():
            os.unlink(temp_audio_path)
    except:
        pass

# Register the cleanup function
import atexit
atexit.register(cleanup)