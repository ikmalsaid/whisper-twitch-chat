import sounddevice as sd
import numpy as np
import queue
import torch
import time
import gc
import threading
import warnings
import tkinter as tk
from datetime import datetime
from collections import deque
from tkinter import ttk, messagebox, scrolledtext
from faster_whisper import WhisperModel, BatchedInferencePipeline
from realtime_chat import RTChatWindow

# Suppress warnings
warnings.filterwarnings("ignore")

class SubtitleWindow:
    def __init__(self):
        # Initialize main window
        self.root = tk.Tk()
        self.root.title("Live Subtitles by Ikmal Said")
        self.root.attributes('-topmost', True)
        self.root.geometry("1100x400+100+100")  # Position at x=100, y=100
        self.root.configure(bg='#18181b')  # Twitch dark theme

        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Configure style for dark theme
        style = ttk.Style()
        style.configure("Chat.Text", background='#0e0e10', foreground='white', font=("Courier New", 12))

        # Create chat display
        self.chat_display = scrolledtext.ScrolledText(
            self.main_frame,
            wrap=tk.WORD,
            bg='#0e0e10',  # Twitch chat background
            fg='white',
            font=("Courier New", 9),
            height=30
        )
        self.chat_display.pack(fill='both', expand=True)
        self.chat_display.tag_configure('timestamp', foreground='#808080')
        self.chat_display.tag_configure('message', foreground='#efeff1')
        self.chat_display.tag_configure('current_line', foreground='#00ff00')  # Green color for current line

        # Bind scroll event to track user scrolling
        self.chat_display.bind('<MouseWheel>', self.on_scroll)
        self.auto_scroll = True  # Flag to control auto-scrolling

        # Status bar at bottom
        self.status_label = ttk.Label(
            self.root,
            text="Status: Loading Whisper model...",
            font=("Courier New", 9),
            background='#18181b',
            foreground='#efeff1'
        )
        self.status_label.pack(side='bottom', fill='x', padx=5, pady=2)

        # Initialize AI Chat Window
        self.ai_chat = RTChatWindow(self.root)

        # Initialize Whisper model
        compute_type = "int8"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize the model with better performance settings
        self.model = WhisperModel(
            model_size_or_path="small",
            device=device,
            compute_type=compute_type,
            cpu_threads=16,  # Adjust based on your CPU
            num_workers=16   # Adjust based on your system
        )
        
        # Create batched inference pipeline
        self.batched_model = BatchedInferencePipeline(
            model=self.model
        )
        
        # Audio processing settings
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.chunk_duration = 1.2  # Increased for better word capture
        self.chunk_samples = int(self.sample_rate * self.chunk_duration)
        
        # Audio buffer settings
        self.audio_buffer = deque(maxlen=6)  # Increased buffer size to 6 seconds
        self.silence_threshold = 0.003  # Reduced to catch quieter speech
        self.min_phrase_time = 0.5
        self.last_process_time = time.time()
        self.processing_delay = 0.001  # Reduced for more frequent updates
        
        # Message history and current message tracking
        self.message_history = []
        self.max_history = 100
        self.current_message = ""
        self.current_message_start = None
        self.word_queue = queue.Queue()
        self.typing_speed = 0.1  # Faster typing speed (milliseconds between words)
        self.last_transcription = ""  # Keep track of last transcription to avoid duplicates
        self.overlap_threshold = 0.15  # 70% overlap threshold for duplicate detection

        # Start the typewriter effect thread
        self.typing_thread = threading.Thread(target=self.typewriter_loop, daemon=True)
        self.typing_thread.start()

        # Update status
        self.update_status("Initializing audio...")

        # Try to initialize audio
        try:
            self.init_audio()
            self.update_status("Ready (Using GPU)" if torch.cuda.is_available() else "Ready (Using CPU)")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize audio: {str(e)}\n\nPlease check your audio input device.")
            self.root.quit()
            return

    def calculate_overlap(self, str1, str2):
        """Calculate the overlap ratio between two strings"""
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        intersection = words1.intersection(words2)
        shorter_len = min(len(words1), len(words2))
        if shorter_len == 0:
            return 0
        return len(intersection) / shorter_len

    def is_duplicate(self, new_text):
        """Check if new text is too similar to last transcription"""
        if not self.last_transcription or not new_text:
            return False
        overlap = self.calculate_overlap(self.last_transcription, new_text)
        return overlap > self.overlap_threshold

    def on_scroll(self, event):
        # Disable auto-scroll if user scrolls up, enable if scrolled to bottom
        if event.delta > 0:  # Scrolling up
            self.auto_scroll = False
        else:  # Scrolling down
            # Check if scrolled to bottom
            if self.chat_display.yview()[1] >= 0.9:
                self.auto_scroll = True

    def ensure_scroll(self):
        if self.auto_scroll:
            self.chat_display.see("end")
            self.chat_display.update_idletasks()

    def typewriter_loop(self):
        while True:
            try:
                timestamp, words = self.word_queue.get()
                if not words:
                    continue

                # Start new line with timestamp
                self.chat_display.configure(state='normal')
                self.chat_display.insert('end', f"[{timestamp}] ", 'timestamp')
                
                # Type each word with a delay
                for i, word in enumerate(words):
                    self.chat_display.insert('end', word + ' ', 'current_line')
                    self.ensure_scroll()
                    time.sleep(self.typing_speed / 1000)  # Convert to seconds
                
                # Add newline after message
                self.chat_display.insert('end', '\n')
                self.chat_display.configure(state='disabled')
                self.ensure_scroll()
                
            except Exception as e:
                print(f"Typewriter error: {str(e)}")
            finally:
                self.word_queue.task_done()

    def add_message(self, text):
        # Check for duplicate/overlapping content
        if self.is_duplicate(text):
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        words = text.split()
        self.word_queue.put((timestamp, words))
        
        # Send to AI chat for reaction
        self.ai_chat.add_subtitle(text)
        
        # Update last transcription
        self.last_transcription = text

        # Add to history
        message = f"[{timestamp}] {text}\n"
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history.pop(0)

    def init_audio(self):
        devices = sd.query_devices()
        input_devices = [i for i, d in enumerate(devices) if d['max_input_channels'] > 0]
        
        if not input_devices:
            raise Exception("No input devices found!")
            
        try:
            default_device = sd.default.device[0]
            if default_device not in input_devices:
                default_device = input_devices[0]
        except:
            default_device = input_devices[0]
            
        print("\nAvailable input devices:")
        for i in input_devices:
            print(f"Device {i}: {devices[i]['name']}")
        print(f"Using device {default_device}: {devices[default_device]['name']}")
        
        self.audio_thread = threading.Thread(target=self.capture_audio, args=(default_device,), daemon=True)
        self.audio_thread.start()

        self.processing_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.processing_thread.start()

    def update_status(self, status):
        self.status_label.config(text=f"Status: {status}")

    def capture_audio(self, device_id):
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio capture status: {status}")
                self.root.after(0, self.update_status, f"Audio capture issue: {status}")
            
            # Add audio data to buffer (ensure it's float32 and memory efficient)
            audio_data = indata.copy().astype(np.float32, copy=False)
            self.audio_buffer.append(audio_data)
            self.audio_queue.put(audio_data)

        try:
            with sd.InputStream(
                device=device_id,
                callback=audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_samples,
                dtype=np.float32  # Explicitly specify dtype
            ):
                self.root.after(0, self.update_status, "Capturing audio...")
                while True:
                    time.sleep(0.1)
        except Exception as e:
            error_msg = f"Audio capture error: {str(e)}"
            print(error_msg)
            self.root.after(0, self.update_status, error_msg)
            self.root.after(0, messagebox.showerror, "Error", error_msg)

    def get_buffered_audio(self):
        if not self.audio_buffer:
            return None
        try:
            # More memory efficient concatenation
            combined = np.concatenate([chunk.reshape(-1, 1) for chunk in self.audio_buffer])
            return combined.reshape(-1)  # Flatten the array
        except Exception as e:
            print(f"Error combining audio: {str(e)}")
            return None

    def process_audio(self):
        while True:
            current_time = time.time()
            
            if len(self.audio_buffer) >= 2 and (current_time - self.last_process_time) >= self.processing_delay:
                try:
                    audio_data = self.get_buffered_audio()
                    
                    if audio_data is not None:
                        # Check if audio data contains any significant sound
                        if np.abs(audio_data).mean() < self.silence_threshold:
                            continue

                        # Process audio with Faster Whisper using batched inference
                        segments, _ = self.batched_model.transcribe(
                            audio_data,
                            batch_size=16,
                            language="en",
                            vad_filter=True,
                            vad_parameters=dict(
                                min_silence_duration_ms=500,
                                speech_pad_ms=100
                            )
                        )

                        # Process each segment
                        for segment in segments:
                            if segment.text.strip():
                                self.root.after(0, self.add_message, segment.text.strip())
                                self.root.after(0, self.update_status, "Speech recognized")
                        
                        # Only clear half the buffer to maintain context
                        for _ in range(len(self.audio_buffer) // 2):
                            if self.audio_buffer:
                                self.audio_buffer.popleft()
                                
                        self.last_process_time = current_time
                        gc.collect()  # Run garbage collection
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()  # Clear CUDA cache if using GPU
                        
                except Exception as e:
                    error_msg = f"Processing error: {str(e)}"
                    print(error_msg)
                    self.root.after(0, self.update_status, error_msg)
                    # Only clear buffer on serious errors
                    if "CUDA out of memory" in str(e):
                        self.audio_buffer.clear()
                        gc.collect()
            
            # time.sleep(0.05)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = SubtitleWindow()
    app.run() 