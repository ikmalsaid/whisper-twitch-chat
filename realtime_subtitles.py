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

class DeviceSelector:
    def __init__(self):
        self.selected_devices = []
        self.window = tk.Tk()
        self.window.title("Audio Device Selection")
        self.window.geometry("600x400")
        self.window.configure(bg='#18181b')
        self.window.resizable(0,0)
        
        # Style configuration
        style = ttk.Style()
        style.configure("Custom.TCheckbutton",
                       background='#18181b',
                       foreground='white')
        
        # Create main frame
        self.frame = ttk.Frame(self.window)
        self.frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Title label
        title_label = ttk.Label(
            self.frame,
            text="Select Audio Input Devices",
            font=("Segoe UI", 12)
        )
        title_label.pack(pady=10)
        
        # Create device list frame with scrollbar
        self.list_frame = ttk.Frame(self.frame)
        self.list_frame.pack(fill='both', expand=True)
        
        # Create canvas for scrolling
        self.canvas = tk.Canvas(self.list_frame, bg='#0e0e10', highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.list_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=560)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Add mousewheel scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Variables to store checkboxes and their variables
        self.device_vars = {}
        self.checkboxes = {}
        
        # Populate devices
        self.populate_devices()
        
        # Buttons
        btn_frame = ttk.Frame(self.frame)
        btn_frame.pack(fill='x', pady=10)
        
        ttk.Button(
            btn_frame,
            text="Confirm Selection",
            command=self.confirm_selection
        ).pack(side='right', padx=5)
        
        ttk.Button(
            btn_frame,
            text="Refresh Devices",
            command=self.populate_devices
        ).pack(side='right', padx=5)
        
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.selection_made = False
    
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def filter_duplicate_devices(self, devices):
        unique_devices = {}
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                name = device['name']
                if name not in unique_devices:
                    unique_devices[name] = (i, device)
        return unique_devices
    
    def populate_devices(self):
        # Clear existing checkboxes
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.device_vars.clear()
        self.checkboxes.clear()
        
        # Get devices
        devices = sd.query_devices()
        self.unique_devices = self.filter_duplicate_devices(devices)
        
        # Create checkboxes for devices
        bg_colors = ['#0e0e10', '#18181b']  # Alternating background colors
        for i, (name, (idx, device)) in enumerate(self.unique_devices.items()):
            # Create frame for each device with alternating background
            device_frame = tk.Frame(
                self.scrollable_frame,
                bg=bg_colors[i % 2],
                padx=5,
                pady=5
            )
            device_frame.pack(fill='x', padx=2, pady=1)
            
            # Create checkbox variable
            var = tk.BooleanVar()
            self.device_vars[idx] = var
            
            # Create checkbox with device info
            checkbox = tk.Checkbutton(
                device_frame,
                text=f"{name} (Channels: {device['max_input_channels']})",
                variable=var,
                bg=bg_colors[i % 2],
                fg='white',
                selectcolor='#2d2d2d',
                activebackground=bg_colors[i % 2],
                activeforeground='white',
                font=("Segoe UI", 10)
            )
            checkbox.pack(side='left', padx=5, pady=2)
            self.checkboxes[idx] = checkbox
    
    def confirm_selection(self):
        # Get selected device indices
        self.selected_devices = [
            idx for idx, var in self.device_vars.items()
            if var.get()
        ]
        
        if not self.selected_devices:
            messagebox.showwarning("Warning", "Please select at least one audio device.")
            return
        
        self.selection_made = True
        self.window.destroy()
    
    def on_closing(self):
        if not self.selection_made:
            if messagebox.askokcancel("Quit", "No devices selected. Do you want to quit?"):
                self.window.destroy()
    
    def get_selected_devices(self):
        self.window.mainloop()
        return self.selected_devices

class AudioCombiner:
    def __init__(self, device_ids, sample_rate, chunk_samples):
        self.device_ids = device_ids
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.streams = []
        self.queues = [queue.Queue() for _ in device_ids]
        self.combined_queue = queue.Queue()
    
    def audio_callback(self, queue_idx):
        def callback(indata, frames, time, status):
            if status:
                print(f"Audio capture status for device {self.device_ids[queue_idx]}: {status}")
            # Ensure the data is 1D by taking the first channel if stereo
            data = indata.copy()
            if data.ndim > 1:
                data = data[:, 0]  # Take first channel if multi-channel
            self.queues[queue_idx].put(data)
        return callback
    
    def start_streams(self):
        for idx, device_id in enumerate(self.device_ids):
            stream = sd.InputStream(
                device=device_id,
                channels=1,  # Force mono channel
                samplerate=self.sample_rate,
                blocksize=self.chunk_samples,
                dtype=np.float32,
                callback=self.audio_callback(idx)
            )
            stream.start()
            self.streams.append(stream)
        
        # Start combination thread
        self.combine_thread = threading.Thread(target=self.combine_audio, daemon=True)
        self.combine_thread.start()
    
    def combine_audio(self):
        while True:
            try:
                # Get audio from all queues
                audio_chunks = []
                for q in self.queues:
                    chunk = q.get(timeout=1.0)
                    # Ensure chunk is 1D
                    if chunk.ndim > 1:
                        chunk = chunk[:, 0]
                    audio_chunks.append(chunk)
                
                # Stack chunks and average them
                if audio_chunks:
                    # Convert to numpy array and ensure 2D shape
                    audio_chunks = np.stack(audio_chunks, axis=0)
                    # Average across devices (axis 0)
                    combined = np.mean(audio_chunks, axis=0)
                    # Ensure the combined audio is 1D
                    combined = combined.flatten()
                    self.combined_queue.put(combined)
                
                # Clear queues to prevent backup
                for q in self.queues:
                    while not q.empty():
                        q.get()
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error combining audio: {e}")
    
    def stop_streams(self):
        for stream in self.streams:
            stream.stop()
            stream.close()
        self.streams.clear()

class SubtitleWindow:
    def __init__(self):
        # First show device selector
        device_selector = DeviceSelector()
        self.selected_devices = device_selector.get_selected_devices()
        
        # Early return if no devices selected
        if not self.selected_devices:
            print("No devices selected. Exiting...")
            self.root = None  # Set root to None to indicate initialization failed
            return
        
        # Print selected devices
        devices = sd.query_devices()
        print("\nSelected devices:")
        for device_id in self.selected_devices:
            print(f"Device {device_id}: {devices[device_id]['name']}")
        
        # Initialize main window
        self.root = tk.Tk()
        self.root.title("whisper-twitch-chat (subtitle) by ikmalsaid")
        self.root.attributes('-topmost', True)
        self.root.geometry("1100x400+100+100")
        self.root.configure(bg='#18181b')

        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Configure style for dark theme
        style = ttk.Style()
        style.configure("Chat.Text", background='#0e0e10', foreground='white', font=("Segoe UI", 12))

        # Create chat display
        self.chat_display = scrolledtext.ScrolledText(
            self.main_frame,
            wrap=tk.WORD,
            bg='#0e0e10',  # Twitch chat background
            fg='white',
            font=("Segoe UI", 9),
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
            font=("Segoe UI", 9),
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
        try:
            # Initialize audio combiner with selected devices
            self.audio_combiner = AudioCombiner(
                device_ids=self.selected_devices,
                sample_rate=self.sample_rate,
                chunk_samples=self.chunk_samples
            )
            
            # Start the streams
            self.audio_combiner.start_streams()
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self.process_audio, daemon=True)
            self.processing_thread.start()
            
            self.update_status("Audio capture started")
            
        except Exception as e:
            error_msg = f"Failed to initialize audio: {str(e)}"
            print(error_msg)
            messagebox.showerror("Error", error_msg)
            self.root.quit()

    def update_status(self, status):
        self.status_label.config(text=f"Status: {status}")

    def process_audio(self):
        while True:
            try:
                # Get combined audio from the combiner
                audio_data = self.audio_combiner.combined_queue.get(timeout=1.0)
                
                # Ensure audio_data is the right shape
                if audio_data is not None:
                    # Ensure audio_data is 1D
                    audio_data = audio_data.flatten()
                    
                    # Add to buffer
                    self.audio_buffer.append(audio_data)
                    
                    current_time = time.time()
                    if len(self.audio_buffer) >= 2 and (current_time - self.last_process_time) >= self.processing_delay:
                        try:
                            # Get buffered audio
                            buffered_audio = np.concatenate(list(self.audio_buffer))
                            
                            # Check if audio data contains any significant sound
                            if np.abs(buffered_audio).mean() < self.silence_threshold:
                                continue

                            # Process audio with Faster Whisper using batched inference
                            segments, _ = self.batched_model.transcribe(
                                buffered_audio,
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
            
            except queue.Empty:
                continue
            except Exception as e:
                error_msg = f"Processing error: {str(e)}"
                print(error_msg)
                self.root.after(0, self.update_status, error_msg)

    def run(self):
        if self.root is not None:  # Only run if initialization was successful
            self.root.mainloop()
        else:
            print("Application not started due to no device selection.")

if __name__ == "__main__":
    app = SubtitleWindow()
    app.run() 