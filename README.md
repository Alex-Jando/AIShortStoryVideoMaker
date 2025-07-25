# AI Short Story Video Maker

An AI-powered, locally run video generator that creates engaging short-form social media videos based on user-provided prompts. This tool combines AI-generated scripts, images, and audio to produce high-quality videos with subtitles.

## Features

- **AI Script Generation**: Automatically generates a structured JSON script for the video, including text and image prompts.
- **AI Image Creation**: Uses Stable Diffusion to generate cartoony AI images for story-related visuals.
- **Real-World Image Search**: Fetches real-world images using the Pexels API for references like locations or objects.
- **Text-to-Speech**: Converts the script into audio using Microsoft Edge TTS with subtitle generation.
- **Video Assembly**: Combines images, audio, and subtitles into a cohesive video with effects.
- **Customizable Prompts**: Allows users to input their own story ideas for unique video content.

## Requirements

To run this project, ensure you have the following installed:

- Python 3.8+
- CUDA-enabled GPU (for Stable Diffusion)
- Required Python libraries (see `requirements.txt`)
- `stabilityai/stable-diffusion-3.5-medium` model installed from hugging face (Other models work as well, just change in `cli.py` or `server.py`)
- `llama3.1` model installed from ollama (Other models work as well, just change in `cli.py` or `server.py`)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/ai-short-story-video-maker.git
   cd ai-short-story-video-maker
   ```
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables for API keys and paths as needed.
Create a `.env` file in the root folder and ensure that the `PEXELS_API_KEY` value is defined.

## Usage

### Command Line

To create a video, run the main script with your desired prompt:

```bash
.\run.bat
...
How many videos would you like to generate? ...
Enter the prompt: ...
Is this the correct prompt? (y/n): y
Generating script...
Script generated.
Generating images...
...
Images generated.
Generating audio...
Audio generated.
Generating video...
...
Video generated and saved as videos/video_1.mp4
```

### Web Server

You can also run the project as a Flask server, allowing users to input prompts via a web interface and receive the generated video when it's complete.

To start the server:
```bash
python server.py
```
By default, the server will be hosted at `http://localhost:8080`. Users can visit this address, enter their prompt, and download the generated video once processing is finished.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.