# whisper-twitch-chat
A fun project that combines live subtitles with live chats like in a real livestream! I have added the ability to mix multiple audio streams (e.g system audio and mic) at the same time.

![Screenshot](assets/thumb.webp)

Powered by:
- OpenAI's Whisper models
- Meta's Llama3.2-1B model
- Faster Whisper
- Ollama (llama.cpp)
- Nvidia CUDA

Installation:
- Clone this repository
- Install `requirements.txt` via `pip`
- Install Ollama and pull the `llama3.2:1b` model
- Launch `realtime_subtitles.py` to start
- Choose the audio streams you want
- Enjoy the live stream and chat!