# app.py
# AI Portrait Stylization — Streamlit app
# Supports two providers:
#  - Replicate (Stable Diffusion img2img)
#  - OpenAI (DALL·E / Image edit)
#
# Usage:
#  - Set REPLICATE_API_TOKEN or OPENAI_API_KEY as environment variables (or paste keys into the UI).
#  - Choose provider in the UI, upload a portrait, select a style preset, then click "Generate".
#
# Note: This app demonstrates the workflow for a final project (education use). Make sure you have the
# appropriate API access and respect terms of service for any model used.

import streamlit as st
from PIL import Image
import io
import os
import base64
import time

# Optional imports for providers
try:
    import replicate
except Exception:
    replicate = None

try:
    import openai
except Exception:
    openai = None

st.set_page_config(page_title="AI Portrait Stylization", layout="centered")

st.title("AI Portrait Stylization — Personal Photo → Artistic Photoshoot")
st.write("Transform your portrait into cinematic / studio / film / magazine styles using Stable Diffusion or DALL·E.")

# Sidebar: provider & API keys
st.sidebar.header("Settings")
provider = st.sidebar.selectbox("Choose inference provider", ["replicate (Stable Diffusion img2img)", "openai (DALL·E / Image edit)"])

if provider.startswith("replicate"):
    replicate_token = st.sidebar.text_input("Replicate API Token (or set REPLICATE_API_TOKEN env)", type="password")
    if not replicate_token:
        replicate_token = os.getenv("REPLICATE_API_TOKEN", "")
else:
    openai_key = st.sidebar.text_input("OpenAI API Key (or set OPENAI_API_KEY env)", type="password")
    if not openai_key:
        openai_key = os.getenv("OPENAI_API_KEY", "")

# Style presets (cinematic / studio / film / magazine)
st.sidebar.markdown("### Style Presets")
style = st.sidebar.selectbox("Select style", ["Cinematic (golden-hour, teal-orange)", "Studio (softbox, high-key)", "Film (grainy, muted tones)", "Magazine (glossy, high-fashion)"])

# Prompt strength / parameters
st.sidebar.markdown("### Generation parameters")
strength = st.sidebar.slider("Style strength (how strongly to apply style)", 0.0, 1.0, 0.6)
num_results = st.sidebar.slider("Number of outputs", 1, 4, 2)
steps = st.sidebar.slider("Inference steps (if supported)", 10, 50, 28)

# Upload image
st.header("Upload a portrait photo")
uploaded_file = st.file_uploader("Upload a face / portrait photo (jpg/png). For best results, use a clear head-and-shoulder photo.", type=["jpg","jpeg","png"])
if uploaded_file:
    input_img = Image.open(uploaded_file).convert("RGB")
    st.image(input_img, caption="Uploaded image (input)", use_column_width=True)

# Compose style-specific prompt templates
def build_prompt(style_choice):
    # Base prompt: keep subject identity but stylize environment/lighting/look
    if "Cinematic" in style_choice:
        base = "A cinematic portrait with golden hour lighting, soft backlight, shallow depth of field, Kodak film grain, warm teal-orange color grade, medium shot, professional retouching"
    elif "Studio" in style_choice:
        base = "A professional studio portrait with softbox lighting, high-key background, crisp skin details, flattering rim light, editorial retouching"
    elif "Film" in style_choice:
        base = "A film-style portrait, muted tones, subtle film grain, natural window light, 35mm film look, moody atmosphere"
    elif "Magazine" in style_choice:
        base = "A glossy magazine editorial portrait, high-fashion pose, dramatic lighting, polished skin, strong composition, studio background"
    else:
        base = "A high-quality artistic portrait, professional lighting and retouching"
    # add guidance about keeping the subject's appearance but applying style
    return f"{base}. Keep the subject's facial features recognizable and natural. High resolution, photorealistic."

# Generate button
if st.button("Generate stylized portraits"):

    if not uploaded_file:
        st.error("Please upload a portrait photo first.")
    else:
        prompt = build_prompt(style)
        # show progress
        status_text = st.empty()
        status_text.info("Starting generation...")

        # Save uploaded image to bytes
        img_bytes = io.BytesIO()
        input_img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        outputs = []
        try:
            if provider.startswith("replicate"):
                if replicate is None:
                    st.error("Replicate library not installed. Please install replicate in your environment.")
                else:
                    token = replicate_token
                    if not token:
                        st.error("Replicate API token missing. Provide it in the sidebar or set REPLICATE_API_TOKEN env var.")
                    else:
                        status_text.info("Calling Replicate (Stable Diffusion img2img)...")
                        client = replicate.Client(api_token=token)
                        # Model choice: use the img2img variant
                        model_name = "stability-ai/stable-diffusion-img2img"
                        # We call replicate.run — note: model input keys may vary by version.
                        # The typical inputs include: prompt, image (image file or URL), strength, num_inference_steps
                        # We will attempt with common input keys. Replicate returns a list of output URLs.
                        for i in range(num_results):
                            status_text.info(f"Generating image {i+1}/{num_results} (Replicate)...")
                            try:
                                output = client.run(model_name, input={
                                    "prompt": prompt,
                                    "image": img_bytes,    # replicate client accepts file-like objects
                                    "strength": float(strength),
                                    "num_inference_steps": int(steps)
                                })
                            except Exception as e:
                                # fallback: try without sending image object directly (upload via bytes URL not available)
                                output = client.run(model_name, input={
                                    "prompt": prompt,
                                    "image": uploaded_file, # try the original uploaded file object
                                    "strength": float(strength),
                                    "num_inference_steps": int(steps)
                                })
                            # output: usually a list of URL(s)
                            if isinstance(output, list) and len(output) > 0:
                                outputs.append(output[0])
                            else:
                                # if replicate returns a single url string
                                outputs.append(output)

            else:
                # OpenAI path
                if openai is None:
                    st.error("OpenAI library not installed. Please install openai in your environment.")
                else:
                    key = openai_key
                    if not key:
                        st.error("OpenAI API key missing. Provide it in the sidebar or set OPENAI_API_KEY env var.")
                    else:
                        status_text.info("Calling OpenAI Image Edit API (DALL·E / GPT-Image)...")
                        openai.api_key = key
                        # Try common edit endpoint: Image.create_edit or images.edit depending on SDK version.
                        for i in range(num_results):
                            status_text.info(f"Generating image {i+1}/{num_results} (OpenAI)...")
                            try:
                                # Many OpenAI SDK versions support Image.create_edit
                                # We'll attempt Image.create_edit first (older SDKs)
                                resp = openai.Image.create_edit(
                                    image=img_bytes,
                                    prompt=prompt,
                                    n=1,
                                    size="1024x1024"
                                )
                                url = resp['data'][0]['url']
                                outputs.append(url)
                            except Exception as e1:
                                try:
                                    # Newer SDKs may use images.generate or images.edit
                                    resp = openai.images.generate(
                                        model="gpt-image-1",
                                        prompt=prompt,
                                        image=img_bytes,
                                        size="1024x1024",
                                        n=1
                                    )
                                    # resp format may vary
                                    if 'data' in resp and len(resp['data'])>0:
                                        if 'b64_json' in resp['data'][0]:
                                            b64 = resp['data'][0]['b64_json']
                                            img_data = base64.b64decode(b64)
                                            # save locally and display
                                            out_path = f"openai_out_{int(time.time())}_{i}.png"
                                            with open(out_path, "wb") as f:
                                                f.write(img_data)
                                            outputs.append(out_path)
                                        elif 'url' in resp['data'][0]:
                                            outputs.append(resp['data'][0]['url'])
                                    else:
                                        outputs.append(None)
                                except Exception as e2:
                                    st.error(f"OpenAI generation error: {e2}")
                                    outputs.append(None)

        except Exception as e:
            st.error(f"Generation failed: {e}")
            status_text.info("Generation stopped due to error.")

        # Display outputs
        status_text.info("Done. Displaying results...")
        if not outputs:
            st.warning("No outputs returned. Check API keys and provider compatibility.")
        else:
            st.header("Generated outputs")
            cols = st.columns(len(outputs))
            for idx, out in enumerate(outputs):
                with cols[idx]:
                    if isinstance(out, str) and (out.startswith("http") or out.endswith(".png") or out.endswith(".jpg") or out.endswith(".jpeg")):
                        try:
                            # if it's a URL
                            if out.startswith("http"):
                                st.image(out, use_column_width=True)
                                st.markdown(f"[Open image in new tab]({out})")
                            else:
                                # local file path
                                img = Image.open(out)
                                st.image(img, use_column_width=True)
                                # provide download link
                                with open(out, "rb") as f:
                                    b = f.read()
                                    href = f'<a href="data:application/octet-stream;base64,{base64.b64encode(b).decode()}" download="stylized_{idx+1}.png">Download</a>'
                                    st.markdown(href, unsafe_allow_html=True)
                        except Exception as e:
                            st.write("Could not display output:", out)
                    else:
                        st.write("Output (raw):", out)

        status_text.success("Generation finished.")
