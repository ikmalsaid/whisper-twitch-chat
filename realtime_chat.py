import tkinter as tk
from tkinter import ttk, scrolledtext
from datetime import datetime
import threading
import queue
import random
import time
import ollama

class RTChatWindow:
    def __init__(self, parent):

        def disable_event(): pass

        self.window = tk.Toplevel(parent)
        self.window.title("Live Chat by Ikmal Said")
        self.window.attributes('-topmost', True)
        self.window.geometry("1100x400")
        self.window.configure(bg='#18181b')  # Twitch dark theme
        self.window.protocol("WM_DELETE_WINDOW", disable_event)

        # Create main frame
        self.main_frame = ttk.Frame(self.window)
        self.main_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Configure style for dark theme
        style = ttk.Style()
        style.configure("Chat.Text", background='#0e0e10', foreground='white', font=("Courier New", 12))

        # Create chat display
        self.chat_display = scrolledtext.ScrolledText(
            self.main_frame,
            wrap=tk.WORD,
            bg='#0e0e10',
            fg='white',
            font=("Courier New", 9),
            height=30
        )
        self.chat_display.pack(fill='both', expand=True)
        
        # Configure text tags for different elements
        self.chat_display.tag_configure('timestamp', foreground='#808080')
        self.chat_display.tag_configure('username', foreground='#FF7F50')  # Coral color
        self.chat_display.tag_configure('message', foreground='#efeff1')
        self.chat_display.tag_configure('system', foreground='#FF4500')  # OrangeRed

        # Message queue and processing
        self.message_queue = queue.Queue()
        self.subtitle_buffer = []  # Will store tuples of (timestamp, text)
        self.last_process_time = time.time()
        self.processing_delay = 0.3  # Reduced from 1.2 to 0.3 seconds
        self.max_buffer_size = 8
        
        # Initialize Ollama
        self.ollama_client = ollama.Client(host='http://localhost:11434')
        
        # Predefined usernames and personalities
        self.usernames = [
            "meme_castle42", "stream_weavr", "ltm_leg3nd", "default_dancer",
            "polyg0nal_picasso", "gg_endgame", "xppowergrind", "pwnz0r_chill",
            "j0yst1ck_j3d1", "qwerty_quester", "neon_dreamz", "glitch_g0ddess"
        ]
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_messages, daemon=True)
        self.processing_thread.start()

        # Auto-scroll
        self.auto_scroll = True
        self.chat_display.bind('<MouseWheel>', self.on_scroll)

        # Position window
        self.window.geometry("+800+100")  # Position the window at x=800, y=100

    def on_scroll(self, event):
        if event.delta > 0:  # Scrolling up
            self.auto_scroll = False
        else:  # Scrolling down
            if self.chat_display.yview()[1] >= 0.9:
                self.auto_scroll = True

    def ensure_scroll(self):
        if self.auto_scroll:
            self.chat_display.see("end")
            self.chat_display.update_idletasks()

    def add_message(self, username, message, message_type='message'):
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        self.chat_display.configure(state='normal')
        self.chat_display.insert('end', f"[{timestamp}] ", 'timestamp')
        self.chat_display.insert('end', f"{username}: ", 'username')
        self.chat_display.insert('end', f"{message}\n", message_type)
        self.chat_display.configure(state='disabled')
        self.ensure_scroll()

    def format_context(self, context_buffer):
        """Format the context buffer into a readable list with timestamps"""
        formatted_lines = []
        for timestamp, text in context_buffer[-self.max_buffer_size:]:
            formatted_lines.append(f"[{timestamp}] {text}")
        return "\n".join(formatted_lines)

    def generate_reaction(self, context):
        try:
            formatted_context = self.format_context(context)
            
            # Prepare the prompt for more Twitch-like reactions
            prompt = f"""You are a live viewer reacting to this part of a live stream:

Recent stream context (newest at bottom):
{formatted_context}

Give a single brief and short reaction that is:
- Highly related to the most recent message
- Casual and conversational
- Engaging but not overly formal
- Must feel natural and spontaneous
- Add emotes ONLY when appropriate

Response should be ONLY the reaction text, nothing else and DO NOT use quotation marks"""

            print("\nContext being processed:")
            print(formatted_context)
            print("--------------------------------")

            response = self.ollama_client.generate(
                model='llama3.2:1b',
                prompt=prompt,
                stream=False,
                options={
                    'temperature': 1.0,
                    'top_p': 0.9,
                    'num_predict': 50,
                }
            )
            return response['response'].strip()
        except Exception as e:
            print(f"Error generating reaction: {str(e)}")
            return None

    def process_messages(self):
        while True:
            try:
                # Get new subtitle if available
                try:
                    subtitle = self.message_queue.get_nowait()
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    self.subtitle_buffer.append((timestamp, subtitle))
                    
                    # Process immediately when new subtitle arrives
                    current_time = time.time()
                    if current_time - self.last_process_time >= self.processing_delay:
                        # Generate 1-2 reactions (reduced from 1-3 for faster responses)
                        num_reactions = random.randint(1, 2)
                        used_usernames = set()
                        
                        for _ in range(num_reactions):
                            reaction = self.generate_reaction(self.subtitle_buffer)
                            if reaction:
                                available_usernames = [u for u in self.usernames if u not in used_usernames]
                                if not available_usernames:
                                    used_usernames.clear()
                                    available_usernames = self.usernames
                                
                                username = random.choice(available_usernames)
                                used_usernames.add(username)
                                
                                # Add reaction to chat using the main thread
                                self.window.after(0, self.add_message, username, reaction)
                                
                                # Reduced delay between messages
                                time.sleep(random.uniform(0.2, 0.5))
                        
                        # Keep only the most recent messages in buffer
                        if len(self.subtitle_buffer) > self.max_buffer_size:
                            self.subtitle_buffer = self.subtitle_buffer[-self.max_buffer_size:]
                        self.last_process_time = current_time

                except queue.Empty:
                    pass

            except Exception as e:
                print(f"Error in message processing: {str(e)}")
            
            time.sleep(0.1)

    def add_subtitle(self, subtitle_text):
        """Add a new subtitle to the processing queue"""
        self.message_queue.put(subtitle_text) 