import streamlit as st
from PIL import Image # To handle image display for the logo
import requests # For making API calls
import os
import google.generativeai as genai # Import the Google Gemini SDK
import base64 # For handling image data from Stability AI if it's base64 encoded
import io # For handling image bytes'
import openai
import base64
from io import BytesIO
from PIL import Image
from io import BytesIO
from lumaai import LumaAI
import time


# --- !!! WARNING: HARDCODING KEYS IS INSECURE !!! ---
# --- !!! REPLACE THE PLACEHOLDER VALUES BELOW WITH YOUR ACTUAL API KEYS !!! ---
GEMINI_API_KEY = "AIzaSyBcrxl3o9NkZLLW49Kzw0cR_wb3g-aaY4M"
ELEVENLABS_API_KEY = "sk_d012d0233baf1aeeb1036165d28327716b06a1b85b05b239"
STABILITY_API_KEY = "sk-tiXOG517rZYqQlHeDHejYfP0VZiCfVwSD1qdI4ykVBwE4M3M" # Replace with your Stability AI API key
OPENAI_API_KEY='sk-4uS4TPLf5oL_EDebgTcU3QONW4YerhTYGFoH9Llfm7T3BlbkFJXeN6bSBC8_a6pv32RqWO3EHbSFDzoxv-BlYOY1idkA'
openai.api_key = OPENAI_API_KEY
# --- !!! END OF API KEY WARNINGS AND CONFIGURATION !!! ---

# --- Stability AI Configuration ---
# You MUST choose a specific engine/model ID for Stability AI.
# Check Stability AI documentation for available and suitable engines.
# Examples: "stable-diffusion-v1-6", "stable-diffusion-xl-1024-v1-0" (for SDXL), etc.
STABILITY_ENGINE_ID = "stable-diffusion-xl-1024-v1-0" # <<< EXAMPLE: CHOOSE YOUR ENGINE
STABILITY_API_HOST = os.getenv('STABILITY_API_HOST', 'https://api.stability.ai')
STABILITY_API_ENDPOINT_TEXT_TO_IMAGE = f"{STABILITY_API_HOST}/v1/generation/{STABILITY_ENGINE_ID}/text-to-image"


# --- Configure Gemini API ---
GEMINI_CONFIGURED = False
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Using a recent and capable model. You can choose other models like 'gemini-pro'.
        gemini_text_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        GEMINI_CONFIGURED = True
        # st.sidebar.success("Gemini API Configured.") # Success messages can be verbose, optionally enable
    except Exception as e:
        st.sidebar.error(f" Config Failed: {e}")
else:
    st.sidebar.warning(" API Key not set. Text generation will use placeholders.")

# --- Configure Stability AI (Basic Check) ---
STABILITY_AI_CONFIGURED = False
if STABILITY_API_KEY and STABILITY_API_KEY != "YOUR_STABILITY_AI_API_KEY_HERE":
    STABILITY_AI_CONFIGURED = True
    # st.sidebar.success("Stability AI API Key Present.") # Optionally enable
else:
    st.sidebar.warning(" AI API Key not set..")


# --- 0. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="KPMG Content Generation Suite",
    page_icon=" KPMG_logo.png" if os.path.exists("KPMG_logo.png") else "🖼️", # Use logo as icon if available
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 1. LOGO AND APPLICATION TITLE ---
try:
    # Ensure your KPMG logo file is in the same directory as your script, or provide the full path.
    logo = Image.open("kpmg_logo.png")
    st.sidebar.image(logo, use_column_width=True)
except FileNotFoundError:
    st.sidebar.warning("KPMG logo (kpmg_logo.png) not found in the script directory.")

st.sidebar.title("KPMG Content Suite") # Shortened title
st.sidebar.markdown("Powered by AI")
st.sidebar.markdown("---") # Visual separator

# --- 2. SIDEBAR NAVIGATION ---
app_mode = st.sidebar.selectbox(
    "Choose a Tool:", # Added colon for clarity
    [
        "Text Generation (Social Posts, Captions, Taglines)",
        "Text-to-Image Creation (Stability AI)",
        "Video Creation (Text/Image to Video with Audio)",
        "Adaptation Generation (Platform-specific Content)",
    ],
    key="app_mode_selector" # Added a key for robustness
)
st.sidebar.markdown("---")
st.sidebar.info( # More user-friendly message
    ""
)


# --- 3. HELPER FUNCTIONS ---

def pil_to_bytes(img, format="PNG"):
    buf = BytesIO()
    img.save(buf, format=format)
    byte_data = buf.getvalue()
    return byte_data

def gemini_text_generation(prompt: str, use_case: str) -> str:
    """
    Generates text using the Gemini API.
    Args:
        prompt (str): The user's input prompt.
        use_case (str): The specific use case (e.g., "social media post").
    Returns:
        str: The generated text or an error/placeholder message.
    """
    if not GEMINI_CONFIGURED:
        return f"Placeholder: Generated {use_case} for: '{prompt}' (Gemini not configured)"
    try:
        # Constructing a more detailed and persona-driven prompt for KPMG
        full_prompt = (
            f"You are an AI assistant for KPMG, a leading global network of professional firms "
            f"providing Audit, Tax, and Advisory services. Your task is to generate content that is "
            f"professional, insightful, clear, concise, and aligns with KPMG's brand values of "
            f"integrity, excellence, courage, togetherness, and for better. "
            f"Avoid overly casual language or jargon where possible, unless appropriate for the specific platform. "
            f"The request is to create a {use_case} based on the following: {prompt}"
        )
        response = gemini_text_model.generate_content(full_prompt)
        # It's good to check if the response has text before accessing it.
        return response.text.strip() if response.text else " returned an empty response."
    except Exception as e:
        st.error(f"Gemini Text Generation Error: {e}")
        return f"Error generating text with Gemini. Details: {str(e)}"
    
    
def luma_video_generation(video_topic: str) -> str:
    
    client = LumaAI(auth_token="luma-79bd1c61-f76b-4d6a-8795-0751876c01cd-82e025a2-3d5b-4a5e-b732-a8f2ec438f1e")

    # st.set_page_config(page_title="LumaAI Video Generator", layout="centered")
    # st.title("🎥 LumaAI Video Generator")
    # st.markdown("Enter a prompt to generate a video using LumaAI's `ray-2` model.")

    # Prompt input
    # prompt = st.text_area("Enter your prompt", value="Create a positive, cheerful, and festive animated scene for New Year's greetings.", height=150)

    # Submit button
    if st.button("Generate Video"):
        with st.spinner("Generating video... please wait ⏳"):
            try:
                # Trigger generation
                generation = client.generations.create(
                    prompt=video_topic,
                    model="ray-2"
                )

                # Polling until done
                while True:
                    generation = client.generations.get(id=generation.id)
                    if generation.state == "completed":
                        break
                    elif generation.state == "failed":
                        st.error(f"Generation failed: {generation.failure_reason}")
                        st.stop()
                    time.sleep(3)

                # Get video
                video_url = generation.assets.video
                video_response = requests.get(video_url)
                video_filename = f"{generation.id}.mp4"
                with open(video_filename, 'wb') as f:
                    f.write(video_response.content)

                st.success("✅ Video generated successfully!")
                st.video(video_filename)

                with open(video_filename, "rb") as file:
                    st.download_button("📥 Download Video", file, file_name=video_filename, mime="video/mp4")

            except Exception as e:
                st.error(f"An error occurred: {e}")
    
def luma_ai_video_generation(prompt: str) -> str:
    """
    Sends an image and prompt to Luma AI and gets a video URL or result.

    Args:
        image (bytes): The image to animate.
        prompt (str): Description of the desired animation or scene.

    Returns:
        str: URL or path to the generated video.
    """
    # --- PLACEHOLDER LOGIC ---
    # Replace this section with actual Luma API calls when available
    st.warning("Luma AI API not available. This is a placeholder.")
    
    # Save uploaded image to simulate sending to Luma
    with open("temp_image_for_luma.png", "wb") as f:
        f.write(image)

    # Simulate video URL
    return "https://www.luma.ai/generated_video_placeholder.mp4"


def stability_ai_image_generation(prompt: str, engine_id: str, height: int = 512, width: int = 512, cfg_scale: float = 7, steps: int = 30, samples: int = 1) -> bytes:
    """
    Generates an image using the Stability AI API.
    Args:
        prompt (str): The text prompt for image generation.
        engine_id (str): The specific Stability AI engine to use.
        height (int): Height of the generated image.
        width (int): Width of the generated image.
        cfg_scale (float): Classifier-Free Guidance scale.
        steps (int): Number of diffusion steps.
        samples (int): Number of images to generate.
    Returns:
        bytes: The generated image bytes, or None if an error occurs.
    """
    if not STABILITY_AI_CONFIGURED:
        st.warning("Stability AI API Key not configured. Returning placeholder.")
        try: # Return bytes of a placeholder image
            placeholder_url = f"https://placehold.co/{width}x{height}/00338D/FFFFFF?text=StabilityAI+Not+Configured"
            response = requests.get(placeholder_url)
            response.raise_for_status()
            return response.content
        except Exception as e:
            st.error(f"Failed to fetch placeholder image: {e}")
            return None

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {STABILITY_API_KEY}",
    }
    # Constructing the payload based on Stability AI documentation
    # The exact payload might vary slightly based on the chosen engine_id
    body = {
        "text_prompts": [{"text": f"KPMG professional style, high quality, corporate visual: {prompt}"}],
        "cfg_scale": cfg_scale,
        "height": height,
        "width": width,
        "samples": samples,
        "steps": steps,
        # Add other parameters like "style_preset", "seed" if supported by the engine and needed
        # e.g., "style_preset": "photographic" for SDXL engines
    }
    # For SDXL models, dimensions like 1024x1024 are common.
    # If using an SDXL engine, adjust default height/width.
    if "xl" in engine_id.lower(): # Basic check if it's an XL model
        body["height"] = 1024
        body["width"] = 1024
        # body["style_preset"] = "photographic" # Example, check Stability AI docs for valid presets

    api_endpoint = f"{STABILITY_API_HOST}/v1/generation/{engine_id}/text-to-image"

    try:
        st.info(f"Sending prompt to Stability AI (Engine: {engine_id}): '{prompt}'")
        api_response = requests.post(api_endpoint, headers=headers, json=body)
        api_response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        response_json = api_response.json()
        # Check for 'artifacts' which is the typical key for image data
        if response_json.get("artifacts") and len(response_json["artifacts"]) > 0:
            image_artifact = response_json["artifacts"][0] # Assuming the first image if multiple are generated
            if image_artifact.get("base64"):
                st.success("Image generated .")
                image_bytes = base64.b64decode(image_artifact["base64"])
                return image_bytes
            else:
                st.error(" No base64 image data found in the artifact.")
                return None
        else:
            st.error(f" No artifacts found in response. Full response: {response_json}")
            return None

    except requests.exceptions.RequestException as e:
        st.error(f"Stability AI API Request Error: {e}")
        if 'api_response' in locals() and api_response is not None: # Check if api_response was defined
            st.error(f"Response status: {api_response.status_code}, content: {api_response.text}")
        return None
    except Exception as e: # Catch any other exceptions
        st.error(f"An unexpected error occurred during  Image Generation: {e}")
        return None

def elevenlabs_text_to_speech(text: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM") -> bytes:
    """
    Converts text to speech using the ElevenLabs API.
    Args:
        text (str): The text to convert.
        voice_id (str): The ID of the ElevenLabs voice to use.
    Returns:
        bytes: The audio data in bytes, or a placeholder/error message.
    """
    if not ELEVENLABS_API_KEY or ELEVENLABS_API_KEY == "YOUR_ELEVENLABS_API_KEY_HERE":
        st.warning("ElevenLabs API Key not configured.")
        return b"Placeholder: Audio (ElevenLabs not configured)" # Return as bytes

    XI_API_KEY = ELEVENLABS_API_KEY
    CHUNK_SIZE = 1024 # Standard chunk size for streaming
    # Example voice ID. You should replace this with a voice ID from your ElevenLabs account.
    # You can list voices via their API: https://api.elevenlabs.io/v1/voices
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "Accept": "audio/mpeg", # Specify expected audio format
        "Content-Type": "application/json",
        "xi-api-key": XI_API_KEY
    }
    # Payload for the API request
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2", # Or another model like "eleven_monolingual_v1"
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            # "style": 0.0, # For style exaggeration, if supported by voice/model
            # "use_speaker_boost": True
        }
    }
    try:
        api_response = requests.post(url, json=data, headers=headers)
        api_response.raise_for_status() # Check for HTTP errors
        
        # Stream the audio content
        audio_bytes = b''
        for chunk in api_response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                audio_bytes += chunk
        return audio_bytes
    except requests.exceptions.RequestException as e:
        st.error(f" API Request Error: {e}")
        if 'api_response' in locals() and api_response is not None:
            st.error(f"Response status: {api_response.status_code}, content: {api_response.text}")
        return None # Indicate failure
    except Exception as e:
        st.error(f"An unexpected error occurred : {e}")
        return None

def generate_video_from_elements(text_prompt: str, image_bytes_list: list = None, audio_bytes: bytes = None) -> str:
    """
    Placeholder function to combine elements into a video.
    Actual implementation would use MoviePy or a similar library.
    Args:
        text_prompt (str): The original text prompt for context.
        image_bytes_list (list): A list of image data in bytes.
        audio_bytes (bytes): Audio data in bytes.
    Returns:
        str: Path or URL to the generated video (currently a placeholder).
    """
    st.info("Video Generation Logic (e.g., using MoviePy) is a placeholder.")
    st.info(f"Received text prompt for video context: {text_prompt[:100]}...")
    if image_bytes_list:
        st.info(f"Received {len(image_bytes_list)} image(s) (as bytes data) for the video.")
    if audio_bytes:
        st.info(f"Received audio data (length: {len(audio_bytes)} bytes) for the video.")

    if audio_bytes and image_bytes_list:
        st.success("Placeholder: Video elements received. Simulating video generation.")
        # TODO: Implement actual video generation using MoviePy
        # 1. Save each item in image_bytes_list to a temporary image file (e.g., .png).
        #    Make sure to handle file naming and cleanup.
        #    Example:
        #    temp_image_paths = []
        #    for i, img_bytes in enumerate(image_bytes_list):
        #        temp_img_path = f"temp_image_{i}.png"
        #        with open(temp_img_path, "wb") as f:
        #            f.write(img_bytes)
        #        temp_image_paths.append(temp_img_path)
        #
        # 2. Save audio_bytes to a temporary audio file (e.g., .mp3).
        #    temp_audio_path = "temp_audio.mp3"
        #    with open(temp_audio_path, "wb") as f:
        #        f.write(audio_bytes)
        #
        # 3. Use MoviePy to:
        #    - Create ImageClips from temp_image_paths.
        #    - Create an AudioFileClip from temp_audio_path.
        #    - Set durations for image clips (e.g., divide audio duration by number of images).
        #    - Concatenate or composite the clips.
        #    - Set the audio of the final video clip.
        #
        # 4. Write the final video to a temporary file (e.g., .mp4).
        #    final_video_path = "generated_video.mp4"
        #    final_clip.write_videofile(final_video_path, fps=24)
        #
        # 5. Return the path to the generated video file (final_video_path).
        #    Don't forget to clean up temporary files after use.
        return "https://www.w3schools.com/html/mov_bbb.mp4" # Placeholder video URL
    else:
        st.warning("Not enough elements to generate video (requires images and audio).")
        return None

def adapt_content_for_platform(content: str, platform: str) -> str:
    """
    Adapts content for a specific platform using Gemini.
    Args:
        content (str): The original content to adapt.
        platform (str): The target platform (e.g., "LinkedIn Post").
    Returns:
        str: The adapted content or an error/placeholder message.
    """
    # The use_case for gemini_text_generation is more descriptive to guide the LLM
    use_case_for_gemini = f"content adaptation for a {platform}"
    # The prompt sent to Gemini includes the original content and the target platform.
    prompt_for_gemini = (
        f"Please adapt the following original content to be perfectly suited for a {platform}. "
        f"Consider the typical audience, tone, length, and formatting conventions of {platform}.\n\n"
        f"Original Content:\n---\n{content}\n---\n\n"
        f"Adapted Content for {platform}:"
    )
    return gemini_text_generation(prompt_for_gemini, use_case_for_gemini)


# --- 4. MAIN APPLICATION SECTIONS ---

# --- 4.1 Text Generation ---
if app_mode == "Text Generation (Social Posts, Captions, Taglines)":
    st.header(" Text Generation")
    st.markdown("Generate compelling text for various marketing needs, maintaining KPMG's professional tone.")

    text_use_case_options = [
        "LinkedIn Post", "Twitter (X) Post (max 280 characters)", "Instagram Caption",
        "Product/Service Tagline (short and catchy)", "Email Subject Line (concise and engaging)",
        "Blog Post Introduction (approx. 100-150 words)", "Executive Summary (brief overview)" 
    ]
    text_use_case = st.selectbox(
        "Select Use Case:", # Added colon
        text_use_case_options,
        key="text_use_case_selector" # Added key
    )
    text_prompt_input = st.text_area(
        "Enter your topic, keywords, or a brief description:",
        height=150,
        key="text_gen_prompt_input", # Added key
        placeholder="e.g., Key insights from our latest cybersecurity report for financial institutions..."
    )

    if st.button("Generate Text", key="text_gen_button", type="primary"): # Added type primary
        if text_prompt_input:
            with st.spinner(f"Generating {text_use_case} ."):
                # The gemini_text_generation function now constructs the full KPMG context prompt
                generated_text = gemini_text_generation(text_prompt_input, text_use_case)
                st.subheader("Generated Text:")
                st.markdown(f"> {generated_text}") # Using blockquote for better visibility
                st.download_button("Download Text", generated_text, file_name=f"{text_use_case.replace(' ', '_')}.txt")
        else:
            st.warning("Please enter a prompt for text generation.")


# --- 4.2 Text-to-Image Creation ---
elif app_mode == "Text-to-Image Creation (Stability AI)":
    st.header("🖼️ Text-to-Image Creation ")
    st.markdown(f" "
                "")
    image_prompt_input = st.text_area(
        "Describe the image you want to create:",
        height=100,
        key="image_gen_prompt_stability", # Unique key
        placeholder="e.g., A diverse team of professionals collaborating in a bright, modern KPMG office..."
    )
    
    images = openai_image_generation(image_prompt_input, n_images=1) # Default to 1 image for simplicity
    
# --- 4.3 Video Creation ---
elif app_mode == "Video Creation (Text/Image to Video with Audio)":
    st.header("🎬 Video Creation for KPMG")
    
    video_topic = st.text_input("Enter video topic/main message:", key="video_topic_input", placeholder="e.g., The future of AI in auditing")
    
    video_path_result = luma_video_generation(video_topic)

# --- 4.4 Adaptation Generation ---
elif app_mode == "Adaptation Generation (Platform-specific Content)":
    st.header("🔄 Content Adaptation with ") # Simplified title
    st.markdown("Tailor existing content for different platforms using Gemini, maintaining KPMG's professional tone.")

    original_content = st.text_area(
        "Paste your original content here:",
        height=200,
        key="adapt_orig_content_input", # Unique key
        placeholder="Enter the content you wish to adapt..."
    )
    target_platform_options = [
        "LinkedIn Post (professional, insightful)", "Twitter (X) Post (concise, engaging, with hashtags)",
        "YouTube Video Description (SEO-friendly, with chapters if applicable)",
        "Internal Company Announcement (clear, direct)", "Website Blog Snippet (engaging summary)"
    ]
    target_platform = st.selectbox(
        "Select Target Platform:", # Added colon
        target_platform_options,
        key="adapt_platform_selector" # Unique key
    )

    if st.button("Adapt Content with AI", key="adapt_gen_button", type="primary"):
        if original_content and target_platform:
            with st.spinner(f"Adapting content for {target_platform.split(' (')[0]} with Gemini..."):
                # The adapt_content_for_platform function handles the specific prompting
                adapted_content = adapt_content_for_platform(original_content, target_platform.split(' (')[0])
                st.subheader(f"Adapted Content for {target_platform.split(' (')[0]}:")
                st.markdown(f"> {adapted_content}")
                st.download_button("Download Adapted Text", adapted_content, file_name=f"adapted_for_{target_platform.split(' (')[0].replace(' ', '_')}.txt")
        else:
            st.warning("Please provide the original content and select a target platform.")

# --- 5. FOOTER (Optional) ---
st.markdown("---") # Visual separator
st.markdown(
    "<div style='text-align: center; color: #555; font-size: 0.9em; padding: 10px;'>" # Slightly styled footer
    "KPMG Content Suite - AI-Powered Content Generation and Adaptation<br>"
    "&copy; KPMG 2025. For internal use and demonstration purposes." # Example copyright
    "</div>",
    unsafe_allow_html=True
)




