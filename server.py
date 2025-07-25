import io
from flask import Flask, jsonify, render_template, request, send_file
import json
import os
import dotenv
import re
import numpy as np
import pysrt
import torch
from diffusers import DiffusionPipeline
import requests
from PIL import Image
from io import BytesIO
import edge_tts
import ollama
from moviepy import AudioFileClip, CompositeVideoClip, ImageClip, TextClip, concatenate_videoclips
from moviepy.video.fx.Resize import Resize

dotenv.load_dotenv(".env")

LANGUAGE_MODEL = "llama3.1"
IMAGE_MODEL = "stabilityai/stable-diffusion-3.5-medium"
VOICE = "en-US-AndrewMultilingualNeural"
VIDEO_RESOLUTION = (720, 1280)

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
FONT_FILE = "res/LuckiestGuy-Regular.ttf"
PROMPT_FILE = "res/prompt.txt"
AUDIO_FILE = "res/temp.mp3"
SRT_FILE = "res/temp.srt"
OUTPUT_DIR = "videos"
SUBTITLE_FONT_SIZE = 64

def generate_script(prompt):

    with open(PROMPT_FILE, "r") as f:
        scriptPrompt = f.read()

    scriptPrompt = scriptPrompt.replace("{PROMPT}", prompt)

    script = json.loads(ollama.generate(model=LANGUAGE_MODEL, prompt=scriptPrompt, format="json").response)["SCRIPT"]

    return script

def resize_image(image, resolution):
    original_width, original_height = image.size

    target_width = original_height * 9 // 16
    target_height = original_height

    if target_width > original_width:
        target_width = original_width
        target_height = original_width * 16 // 9

    left = (original_width - target_width) // 2
    top = (original_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height

    img_cropped = image.crop((left, top, right, bottom))

    return img_cropped.resize((resolution[0], resolution[1]))

def generate_image(prompt):

    # Setting guidance scale to 12.0 creates an image which focuses on the prompt more
    # Setting guidance scale to 7.5 creates an image which focuses on the prompt less

    # Setting num_inference_steps to 28 creates an image with sufficient quality and detail

    image = pipe(prompt, guidance_scale=7.5, num_inference_steps=28, width=576, height=1024).images[0]

    image = resize_image(image, VIDEO_RESOLUTION)

    return image

def download_image(query):

    response = requests.get("https://api.pexels.com/v1/search",
                            params={"query": query},
                            headers={"Authorization": PEXELS_API_KEY})

    response = response.json()
    photos = response["photos"]

    for photo in photos:
        if photo["width"] > VIDEO_RESOLUTION[0] and photo["height"] > VIDEO_RESOLUTION[1]:
            image_data = requests.get(photo["src"]["original"]).content
            img = Image.open(BytesIO(image_data))

            return resize_image(img, VIDEO_RESOLUTION)
        
def generate_audio(text, voice, audio_file, srt_file):
    communicate = edge_tts.Communicate(text, voice, rate="+0%", volume="-10%")
    submaker = edge_tts.SubMaker()
    with open(audio_file, "wb") as file:
        for chunk in communicate.stream_sync():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                submaker.feed(chunk)
    with open(srt_file, "w", encoding="utf-8") as file:
        file.write(submaker.get_srt())

def calculate_image_timings(srt_file, text, images):

    subs = pysrt.open(srt_file)
    
    image_durations = []

    sentence_index = 0
    sentence_words = []
    sentence_start_time = 0
    sentence_end_time = None

    punctuation_re = re.compile(r'[^a-zA-Z0-9\s]')

    for sub in subs:

        if sentence_index >= len(text):
            break

        word = sub.text.strip()

        sentence_words.append(word)
        
        sentence_text = " ".join(sentence_words).strip()

        sentence_text_cleaned = punctuation_re.sub('', sentence_text).strip().casefold()
        sentence_text_cleaned = re.sub(r'\s+', ' ', sentence_text_cleaned).strip()

        sentence_text_from_array = punctuation_re.sub('', text[sentence_index]).strip().casefold()
        sentence_text_from_array = re.sub(r'\s+', ' ', sentence_text_from_array).strip()

        if sentence_text_cleaned == sentence_text_from_array:
     
            sentence_end_time = sub.end.ordinal
            
            duration = (sentence_end_time - sentence_start_time) / 1000.0
            
            image_durations.append((images[sentence_index], duration))
            
            sentence_words = []
            sentence_start_time = sentence_end_time
            sentence_index += 1

    return image_durations

def apply_effect(clip):

    return clip.with_effects([Resize(lambda t: 1 + 0.04 * t)])

def create_video_with_effects(image_timings, output_path):
    clips = []

    for pil_image, duration in image_timings:

        clip = ImageClip(np.array(pil_image), duration=duration)

        clip = apply_effect(clip)

        clips.append(clip)

    video = concatenate_videoclips(clips, method="compose")

    video = video.with_audio(AudioFileClip(AUDIO_FILE))

    subs = pysrt.open(SRT_FILE)

    subtitle_clips = []

    i = 0

    while i < len(subs):

        if i + 5 < len(subs):

            buffer_subs = subs[i : i + 5]

        else:

            buffer_subs = subs[i :]

        text_clip = TextClip(
                    text=" ".join([sub.text for sub in buffer_subs]),
                    font=FONT_FILE,
                    font_size=SUBTITLE_FONT_SIZE,
                    color="yellow",
                    stroke_color="black",
                    stroke_width=2,
                    method="caption",
                    size=(VIDEO_RESOLUTION[0] * 4 // 5, SUBTITLE_FONT_SIZE * 4),
                    text_align="center",
                    vertical_align="center",
                    horizontal_align="center"
                ).with_position(("center", VIDEO_RESOLUTION[1] * 2 // 3))

        text_clip = text_clip.with_position(("center", VIDEO_RESOLUTION[1] * 2 // 3))

        text_clip = text_clip.with_start(buffer_subs[0].start.ordinal / 1000)

        text_clip = text_clip.with_end(buffer_subs[-1].end.ordinal / 1000)

        subtitle_clips.append(text_clip)

        i += 5

    subtitle_video = CompositeVideoClip([video] + subtitle_clips)

    subtitle_video.write_videofile(output_path, fps=24, codec="h264_nvenc", threads=12, preset="fast")

    subtitle_video.close()

def generate_video(prompt) -> str:

        print("Generating script...")

        script = None
        text = []
        image_queries = []
        validJson = False

        while not validJson:
            try:
                script = generate_script(prompt)

                text = [t["TEXT"] for t in script if "TEXT" in t.keys()]

                image_queries = []

                for i in script:
                    if "AI_IMAGE" in i.keys() or "SEARCH_IMAGE" in i.keys():
                        image_queries.append(i)

                if len(text) != len(image_queries):
                    continue

                validJson = True
            except json.JSONDecodeError:
                continue

        print("Script generated.")

        print("Generating images...")

        images = []

        for i, query in enumerate(image_queries):
            if "AI_IMAGE" in query.keys():
                images.append(generate_image(query["AI_IMAGE"]))
            elif "SEARCH_IMAGE" in query.keys():
                images.append(download_image(query["SEARCH_IMAGE"]))
            print(f"Image {i + 1} generated.")

        print("Images generated.")

        # Generate the audio and subtitle files
        print("Generating audio...")
        generate_audio(" ".join(text), VOICE, AUDIO_FILE, SRT_FILE)
        print("Audio generated.")

        # Calculate the image timings
        image_timings = calculate_image_timings(SRT_FILE, text, images)

        print("Generating video...")

        os.listdir(OUTPUT_DIR)

        i = 1

        while f"video_{i}.mp4" in os.listdir(OUTPUT_DIR):
            i += 1

        create_video_with_effects(image_timings, f"{OUTPUT_DIR}/video_{i}.mp4")

        os.remove(AUDIO_FILE)
        os.remove(SRT_FILE)

        print(f"Video generated and saved as {OUTPUT_DIR}/video_{i}.mp4")

        return f"{OUTPUT_DIR}/video_{i}.mp4"

def main():

    prompts = []
    video_number = None

    while True:
        try:
            video_number = int(input("How many videos would you like to generate? "))
            break
        except ValueError:
            print("Please enter a valid number.")

    for _ in range(video_number):
        while True:
            prompt = input("Enter the prompt: ")
            if input("Is this the correct prompt? (y/n): ").casefold() == "y".casefold():
                prompts.append(prompt)
                break

    for prompt in prompts:
        generate_video(prompt)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')

    if not prompt:

        return jsonify({'error': 'Prompt is required'}), 400

    try:

        video_path = generate_video(prompt)

        with open(video_path, 'rb') as f:
            response = send_file(io.BytesIO(f.read()), mimetype='video/mp4', as_attachment=True, download_name='video.mp4')

        os.remove(video_path)

        return response
    
    except Exception as e:

        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    pipe = DiffusionPipeline.from_pretrained(IMAGE_MODEL, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    app.run(host="0.0.0.0", port=8080)