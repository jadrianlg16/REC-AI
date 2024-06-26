'''
todo: fix microphone and system audio implementations
attempting to integrate recording of system audio, and integrate that recording into my system for transcribing audio to text


works on laptop not on pc maybe because stero/microphone and such


'''


import tkinter as tk
from tkinter import ttk  # for improved styling
from tkinter import filedialog, Label, Entry, Button, StringVar, OptionMenu, DISABLED, NORMAL
import requests
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from pydub import AudioSegment
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import os
import time
import re  # Add this import at the top of your script
import pyaudio
import threading
import wave  # For saving the recorded audio to a WAV file
import sounddevice as sd
import numpy as np




def extract_and_convert_audio(video_path, language="es-ES", chunk_length_ms=60000):
    start_time = time.time()  # Start the timer for the entire process

    # Extract audio from video
    video = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path)
    video.close()

    # Load extracted audio
    audio = AudioSegment.from_wav(audio_path)
    chunks = split_audio(audio, chunk_length_ms)

    # Initialize speech recognizer
    recognizer = sr.Recognizer()
    transcript = ""

    # Process each chunk
    for i, chunk in enumerate(chunks):
        chunk_start_time = time.time()  # Start the timer for this chunk
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        chunk_file = "temp_chunk.wav"
        chunk.export(chunk_file, format="wav")
        try:
            with sr.AudioFile(chunk_file) as source:
                audio_listened = recognizer.record(source)
                # Try converting it to text
                try:
                    text = recognizer.recognize_google(audio_listened, language=language)
                    transcript += text + " "
                except sr.UnknownValueError as e:
                    print(f"Chunk {i+1} could not be understood.")
                except sr.RequestError as e:
                    print(f"API request error on chunk {i+1}: {str(e)}")
        finally:
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
        chunk_end_time = time.time()  # End the timer for this chunk
        print(f"Chunk {i+1} processed in {chunk_end_time - chunk_start_time:.2f} seconds.")

    save_transcript(video_path, transcript)
    if os.path.exists(audio_path):
        os.remove(audio_path)

    end_time = time.time() 
    print(f"Total processing time: {end_time - start_time:.2f} seconds.")

def split_audio(audio_segment, chunk_length_ms):
    """Splits the audio into chunks of a specific length."""
    return [audio_segment[i:i + chunk_length_ms] for i in range(0, len(audio_segment), chunk_length_ms)]

def save_transcript(video_path, transcript):
    """Saves the transcript to a file named after the video."""
    base = os.path.splitext(os.path.basename(video_path))[0]
    transcript_path = os.path.join(os.path.dirname(video_path), f"{base}_transcript.txt")
    with open(transcript_path, "w", encoding="utf-8") as file:
        file.write(transcript)
    print(f"Transcript saved to {transcript_path}")

def fetch_youtube_transcript(video_url: str) -> str:
    try:
        # Use regular expressions to find video IDs in various YouTube URL formats
        video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', video_url)
        if not video_id_match:
            print("No valid video ID found in URL.")
            return ""

        video_id = video_id_match.group(1)
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        formatter = TextFormatter()
        transcript = formatter.format_transcript(transcript_list)
        return transcript
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""



class TranscriptionApp:
    
    def __init__(self, master):
        self.master = master
        master.title("Video to Transcript App")

        # Set the overall font for the application
        default_font = ('Segoe UI', 10)
        master.option_add("*Font", default_font)

        # Apply a dark theme to the application for better aesthetics
        self.style = ttk.Style(master)
        self.style.theme_use('clam')  # 'clam' is a modern-looking theme

        # Initialize style configurations
        self.configure_styles()

        # Configure the grid layout
        self.configure_grid()

        # Initialize variables
        self.tr_service_var = StringVar(master)
        self.ai_service_var = StringVar(master)
        self.model_var = StringVar(master)
        self.language_var = StringVar(master)
        self.initialize_variables()



        # Initialize the languages dictionary
        self.languages = {
            "English": "en-US",
            "Spanish": "es-ES",
            "French": "fr-FR"
            # Add more languages as needed
        }

        self.is_recording = False
        self.is_recording_system_audio = False

        # Create frames for different sections of the GUI
        self.create_frames()

        # Create widgets for each section
        self.create_left_frame_widgets()
        self.create_right_frame_widgets()
        self.create_bottom_frame_widgets()

        # Bind events for dynamic updates based on user interaction
        self.bind_events()

        # Set the initial states for widgets that depend on other widgets
        self.update_widgets_state()

    def configure_styles(self):
        # Theme
        self.style.theme_use('clam')
        
        # Frame style
                # Assuming 'darkBackground' is your chosen background color
        darkBackground = '#2D2D2D'
        self.style.configure('TFrame', background=darkBackground)
        
        # Label styles
        self.style.configure('TLabel', background='#2D2D2D', foreground='#CCCCCC', font=('Segoe UI', 10))
        
        # Entry styles
        # self.style.configure('TEntry', foreground='#FFFFFF', fieldbackground='#333333', font=('Segoe UI', 10), borderwidth = 0)
        self.style.configure('TEntry', foreground='#FFFFFF', fieldbackground='#333333', font=('Segoe UI', 10), borderwidth=0, highlightthickness=0)


        # Button styles
        self.style.configure('TButton', font=('Segoe UI', 10), borderwidth=0, background='#4E4E4E', foreground='#FFFFFF')
        self.style.map('TButton',
                    foreground=[('!disabled', '#FFFFFF'), ('active', '#FFFFFF'), ('pressed', '#FFFFFF')],
                    background=[('!disabled', '#4E4E4E'), ('active', '#5E5E5E'), ('pressed', '#333333')])
        
        # Combobox styles
        self.style.configure('TCombobox', fieldbackground='#333333',borderwidth=0, background='#2D2D2D', foreground='#CCCCCC', arrowcolor='#FFFFFF')
        self.style.map('TCombobox',
                    fieldbackground=[('!disabled', '#333333'), ('active', '#333333')],
                    background=[('!disabled', '#2D2D2D'), ('active', '#2D2D2D')],
                    foreground=[('!disabled', '#CCCCCC'), ('active', '#CCCCCC')],
                    arrowcolor=[('!disabled', '#CCCCCC'), ('active', '#FFFFFF')])

        # Special Button styles for emphasis
        self.style.configure('Special.TButton', font=('Segoe UI', 10), borderwidth=0, background='#0078D7', foreground='#FFFFFF')
        self.style.map('Special.TButton',
                    foreground=[('!disabled', '#FFFFFF'), ('active', '#FFFFFF'), ('pressed', '#FFFFFF')],
                    background=[('!disabled', '#0078D7'), ('active', '#005A9E'), ('pressed', '#003A6F')])

        # Optionally, configure additional widget styles as needed


# Configuiring UI

    def configure_grid(self):
        self.master.grid_rowconfigure(0, weight=0)
        self.master.grid_columnconfigure(0, weight=0)
        self.master.grid_columnconfigure(1, weight=0)

    def initialize_variables(self):
        # Initialize any tkinter variables and set default values here
        self.tr_service_var.set("Youtube")  # Example default setting
        self.ai_service_var.set("OpenAI")  # Example default setting

    def create_frames(self):
        self.left_frame = ttk.Frame(self.master, padding="10", borderwidth="1")
        self.left_frame.grid(row=0, column=0, sticky='nsew', padx=(10, 5), pady=10)

        self.right_frame = ttk.Frame(self.master, padding="10")
        self.right_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 10), pady=10)

        self.bottom_frame = ttk.Frame(self.master, padding="10")
        self.bottom_frame.grid(row=1, column=0, columnspan=2, sticky='ew', padx=(5, 10), pady=10)



    def create_left_frame_widgets(self):
        padding = {'padx': 5, 'pady': 5}
        self.label_tr_service = ttk.Label(self.left_frame, text="Transcript Service:")
        self.label_tr_service.grid(row=0, column=0, sticky='e', **padding)
        self.tr_service_menu = ttk.Combobox(self.left_frame, textvariable=self.tr_service_var, 
                                            values=["Youtube", "Local Video", "Record my Audio"], state="readonly")
        self.tr_service_menu.grid(row=0, column=1, sticky='ew', **padding)

        self.label_video_path = ttk.Label(self.left_frame, text="Video Path:")
        self.label_video_path.grid(row=1, column=0, sticky='e', **padding)
        self.entry_video_path = ttk.Entry(self.left_frame, width=50)
        self.entry_video_path.grid(row=1, column=1, sticky='ew', **padding)
        self.button_browse_video = ttk.Button(self.left_frame, text="Browse", command=self.browse_video)
        self.button_browse_video.grid(row=1, column=2, **padding)

        self.label_youtube_url = ttk.Label(self.left_frame, text="YouTube Video URL:")
        self.label_youtube_url.grid(row=2, column=0, sticky='e', **padding)
        self.entry_youtube_url = ttk.Entry(self.left_frame, width=50)
        self.entry_youtube_url.grid(row=2, column=1, sticky='ew', **padding)
        self.button_fetch_transcript = ttk.Button(self.left_frame, text="Fetch Transcript", command=self.fetch_and_display_transcript)
        self.button_fetch_transcript.grid(row=2, column=2, **padding)

        self.button_start_transcription = ttk.Button(self.left_frame, text="Start Transcription", command=self.start_transcription, style='Special.TButton')
        self.button_start_transcription.grid(row=3, column=0, columnspan=3, sticky='ew', **padding)

        # Assuming these are placed within a method that sets up widgets for the audio recording option
        self.record_microphone_button = ttk.Button(self.left_frame, text="Record Microphone Audio", command=self.record_microphone_audio)
        self.record_microphone_button.grid(row=4, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
        self.record_microphone_button.grid_remove()  # Initially hide this button

        self.record_system_audio_button = ttk.Button(self.left_frame, text="Record System Audio", command=self.start_system_audio_recording)
        self.record_system_audio_button.grid(row=5, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
        self.record_system_audio_button.grid_remove()  # Initially hide this button

        # Add a stop button in your GUI setup
        self.stop_recording_button = ttk.Button(self.left_frame, text="Stop Recording", command=self.stop_recording)
        self.stop_recording_button.grid(row=6, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
        self.stop_recording_button.grid_remove()  # Hide it initially

        self.stop_recordingSA_button = ttk.Button(self.left_frame, text="Stop RecordingSA", command=self.stop_system_audio_recording)
        self.stop_recordingSA_button.grid(row=7, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
        self.stop_recordingSA_button.grid_remove()  # Hide it initially

    def create_right_frame_widgets(self):
        padding = {'padx': 5, 'pady': 5}
        # AI Service Selection
        self.label_ai_service = ttk.Label(self.right_frame, text="AI Service:")
        self.label_ai_service.grid(row=0, column=0, sticky='e',**padding)
        self.ai_service_menu = ttk.Combobox(self.right_frame, textvariable=self.ai_service_var, 
                                            values=["OpenAI", "Local LLM"], state="readonly")
        self.ai_service_menu.grid(row=0, column=1, sticky='ew',**padding)

        # API Key Entry
        self.label_api_key = ttk.Label(self.right_frame, text="API Key:")
        self.label_api_key.grid(row=1, column=0, sticky='e',**padding)
        self.entry_api_key = ttk.Entry(self.right_frame, width=50)
        self.entry_api_key.grid(row=1, column=1, sticky='ew',**padding)

        # Model Selection Combobox
        self.label_model = ttk.Label(self.right_frame, text="Model:")
        self.label_model.grid(row=2, column=0, sticky='e',**padding)
        self.model_menu = ttk.Combobox(self.right_frame, textvariable=self.model_var, 
                                    values=["GPT-4", "GPT-3.5 Turbo"], state="readonly")
        self.model_menu.grid(row=2, column=1, sticky='ew',**padding)

        # Local LLM URL Entry
        self.label_llm_url = ttk.Label(self.right_frame, text="Local LLM URL:")
        self.label_llm_url.grid(row=3, column=0, sticky='e',**padding)
        self.entry_llm_url = ttk.Entry(self.right_frame, width=50)
        self.entry_llm_url.grid(row=3, column=1, sticky='ew',**padding)

        # Prompt Entry
        self.label_prompt = ttk.Label(self.right_frame, text="Prompt:")
        self.label_prompt.grid(row=4, column=0, sticky='e',**padding)
        self.entry_prompt = ttk.Entry(self.right_frame, width=50)
        self.entry_prompt.grid(row=4, column=1, sticky='ew',**padding)

        # System Message for Local LLM Entry
        self.label_system_message = ttk.Label(self.right_frame, text="System Message for Local LLM:")
        self.label_system_message.grid(row=5, column=0, sticky='e',**padding)
        self.entry_system_message = ttk.Entry(self.right_frame, width=50)
        self.entry_system_message.grid(row=5, column=1, sticky='ew',**padding)

        # Save Response Path Entry and Browse Button
        self.label_save_path = ttk.Label(self.right_frame, text="Save Response Path:")
        self.label_save_path.grid(row=6, column=0, sticky='e',**padding)
        self.entry_save_path = ttk.Entry(self.right_frame, width=50)
        self.entry_save_path.grid(row=6, column=1, sticky='ew',**padding)
        self.button_browse_save_path = ttk.Button(self.right_frame, text="Browse", command=self.browse_save_path)
        self.button_browse_save_path.grid(row=6, column=2)

        # Language/Model Code Dropdown
        self.label_language = ttk.Label(self.right_frame, text="Language/Model Code:")
        self.label_language.grid(row=7, column=0, sticky='e',**padding)
        self.language_menu = ttk.Combobox(self.right_frame, textvariable=self.language_var, 
                                        values=list(self.languages.keys()), state="readonly")
        self.language_menu.grid(row=7, column=1, sticky='ew',**padding)
        self.language_menu.set("English")  # Set default value

        # Send Prompt Button
        self.button_send_prompt = ttk.Button(self.right_frame, text="Send Prompt", command=self.send_to_api, style='Special.TButton')
        self.button_send_prompt.grid(row=8, column=0, columnspan=3, sticky='ew',**padding)


    def create_bottom_frame_widgets(self):
        # Your bottom frame widget creation code here
        self.system_message_label = ttk.Label(self.bottom_frame, text="Ready to start transcription.")
        self.system_message_label.grid(row=0, column=0, sticky='w')

    def bind_events(self):
        self.tr_service_menu.bind('<<ComboboxSelected>>', self.toggle_transcript_options)
        self.ai_service_menu.bind('<<ComboboxSelected>>', self.toggle_api_key_url)

    
    def update_widgets_state(self):
        # Update widgets that depend on the AI service selection
        self.toggle_api_key_url()
        # Update widgets that depend on the transcript service selection
        self.toggle_transcript_options()
        # Reset system message
        self.system_message_label['text'] = "Ready."





#Toggoling UI Code************************************

    def toggle_api_key_url(self, *args):
        if self.ai_service_var.get() == "OpenAI":
            self.show_openai_widgets()
            self.hide_llm_widgets()
        elif self.ai_service_var.get() == "Local LLM":
            self.hide_openai_widgets()
            self.show_llm_widgets()

    def show_openai_widgets(self):
        self.entry_api_key.grid()
        self.label_api_key.grid()
        self.entry_prompt.grid()
        self.label_prompt.grid()
        self.model_menu.grid()
        self.label_model.grid()

    def hide_openai_widgets(self):
        self.entry_api_key.grid_remove()
        self.label_api_key.grid_remove()
        self.model_menu.grid_remove()
        self.label_model.grid_remove()

    def show_llm_widgets(self):
        self.entry_llm_url.grid()
        self.label_llm_url.grid()
        self.entry_system_message.grid()
        self.label_system_message.grid()

    def hide_llm_widgets(self):
        self.entry_llm_url.grid_remove()
        self.label_llm_url.grid_remove()
        self.entry_system_message.grid_remove()
        self.label_system_message.grid_remove()


  
    def toggle_transcript_options(self, *args):
        selected_service = self.tr_service_var.get()

        if selected_service == "Youtube":
            self.show_youtube_widgets()
            self.hide_local_video_widgets()
            self.hide_audio_recording_widgets()
        elif selected_service == "Local Video":
            self.hide_youtube_widgets()
            self.show_local_video_widgets()
            self.hide_audio_recording_widgets()
        elif selected_service == "Record my Audio":
            self.hide_youtube_widgets()
            self.hide_local_video_widgets()
            self.show_audio_recording_widgets()

    def show_youtube_widgets(self):
        self.entry_youtube_url.grid()
        self.label_youtube_url.grid()
        self.button_fetch_transcript.grid()

    def hide_youtube_widgets(self):
        self.entry_youtube_url.grid_remove()
        self.label_youtube_url.grid_remove()
        self.button_fetch_transcript.grid_remove()

    def show_local_video_widgets(self):
        self.entry_video_path.grid()
        self.label_video_path.grid()
        self.button_browse_video.grid()

    def hide_local_video_widgets(self):
        self.entry_video_path.grid_remove()
        self.label_video_path.grid_remove()
        self.button_browse_video.grid_remove()

    def show_audio_recording_widgets(self):
        self.record_microphone_button.grid()  # Show the microphone recording button
        self.record_system_audio_button.grid()  # Show the system audio recording button (functionality not implemented yet)
        self.stop_recordingSA_button.grid()
    def hide_audio_recording_widgets(self):
        self.record_microphone_button.grid_remove()
        self.record_system_audio_button.grid_remove()


#End of toggoling UI Code************************************
    def record_microphone_audio(self):
        """Start recording audio from the microphone."""
        self.is_recording = True
        self.stop_recording_button.grid()  # Show the stop recording button

        # Start recording in a separate thread to prevent blocking the GUI
        threading.Thread(target=self._record_audio_thread, daemon=True).start()

    def _record_audio_thread(self):
        """Method that runs in a separate thread to record audio."""
        chunk = 1024  # Record in chunks of 1024 samples
        sample_format = pyaudio.paInt16  # 16 bits per sample
        channels = 2
        fs = 44100  # Record at 44100 samples per second

        p = pyaudio.PyAudio()  # Create an interface to PortAudio

        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)

        frames = []  # Initialize array to store frames

        # Keep recording until is_recording is set to False
        while self.is_recording:
            data = stream.read(chunk)
            frames.append(data)

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        # Terminate the PortAudio interface
        p.terminate()

        # Save the recorded data as a WAV file
        self.save_recorded_audio(frames, fs, channels, sample_format)

        self.stop_recording_button.grid_remove()  # Hide the stop button again

    def save_recorded_audio(self, frames, fs, channels, sample_format):
        """Saves the recorded audio to a WAV file and transcribes it."""
        filename = "recorded_audio.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()

        # Now that we have an audio file, you can transcribe it using your existing process
        # Adjust this to use the correct method for transcribing an audio file
        self.transcribe_audio_file(filename)  # Implement this method

    def stop_recording(self):
        """Stops recording audio from the microphone."""
        self.is_recording = False


    def record_system_audio(self, filename="system_audio.wav"):
        """Record system audio until stopped."""
        samplerate = 44100  # Sample rate
        channels = 2  # Stereo

        # Initialize a buffer to hold chunks of audio data
        audio_buffer = []

        def callback(indata, frames, time, status):
            """This is called (from a separate thread) for each audio block."""
            if status:
                print(status)
            audio_buffer.append(indata.copy())

        # Open a new stream for recording
        with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback, dtype='int16'):
            self.is_recording_system_audio = True
            print("Recording system audio...")
            while self.is_recording_system_audio:
                sd.sleep(100)  # Sleep for a short time to prevent high CPU usage
            print("Recording finished")

        # Convert the list of numpy arrays into a single numpy array
        audio = np.concatenate(audio_buffer, axis=0)

        # Save the recorded audio to a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # Assuming 16-bit audio
        wf.setframerate(samplerate)
        wf.writeframes(audio.astype(np.int16).tobytes())
        wf.close()
        print(f"System audio saved to {filename}")

    def start_system_audio_recording(self):
        """Start recording system audio."""
        threading.Thread(target=self.record_system_audio, args=("system_audio.wav",), daemon=True).start()

    def stop_system_audio_recording(self):
        """Stop recording system audio."""
        self.is_recording_system_audio = False

    def fetch_and_display_transcript(self):
        video_url = self.entry_youtube_url.get()
        video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', video_url)
        if video_id_match:
            video_id = video_id_match.group(1)
            transcript = fetch_youtube_transcript(video_url)
            if transcript:
                # Save the transcript to a file
                transcript_path = os.path.join(os.path.dirname(__file__), f"{video_id}_transcript.txt")
                with open(transcript_path, "w", encoding="utf-8") as file:
                    file.write(transcript)
                print(f"Transcript saved to {transcript_path}")
                # Set the transcript_path to entry_video_path for further processing
                self.entry_video_path.delete(0, tk.END)  # Clear the entry field
                self.entry_video_path.insert(0, transcript_path)  # Insert the path of the saved transcript
            else:
                print("Failed to fetch transcript.")
        else:
            print("Invalid YouTube URL provided.")

    def browse_video(self):
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(("MP4 Files", "*.mp4"), ("All Files", "*.*"))
        )
        if video_path:  # If a file is selected, update the path in the entry.
            self.entry_video_path.delete(0, tk.END)
            self.entry_video_path.insert(0, video_path)

    def start_transcription(self):
        video_path = self.entry_video_path.get()
        language_code = self.languages.get(self.language_var.get(), "en-US")  # Default to "en-US" if not found
        
        if video_path:
            self.system_message_label['text'] = "Starting transcription..."
            extract_and_convert_audio(video_path, language_code)
            self.system_message_label['text'] = "Transcription completed."
        else:
            print("No video selected")

    def browse_save_path(self):
        save_path = filedialog.asksaveasfilename(
            title="Select Save Response Path",
            defaultextension=".txt",
            filetypes=(("Text Files", "*.txt"), ("All Files", "*.*"))
        )
        if save_path:  # If a path is selected, update the path in the entry.
            self.entry_save_path.delete(0, tk.END)
            self.entry_save_path.insert(0, save_path)

    def send_to_api(self):
        service = self.ai_service_var.get()
        transcript_path = self.entry_video_path.get().replace(".mp4", "_transcript.txt")  # This path might be to a transcript or a video file
        
        print(f"Transcript path: {transcript_path}")

        # Determine if we're working with a direct transcript file or need to generate one
        transcript = ""

        if transcript_path.endswith("_transcript.txt") and os.path.exists(transcript_path):
            with open(transcript_path, 'r', encoding='utf-8') as file:
                transcript = file.read()
        else:
            print("Transcript path does not point to a valid transcript file.")
            return  # Exit if no valid transcript file is found

        if service == "OpenAI":
            api_key = self.entry_api_key.get()
            model = self.model_var.get()  # Assuming this variable holds the correct model name
            prompt = self.entry_prompt.get()
            save_path = self.entry_save_path.get()

            if api_key and model:
                self.system_message_label['text'] = "Sending to API..."
                headers = {'Authorization': f'Bearer {api_key}'}
                data = {
                    'model': model,
                    'prompt': f"{prompt}\n\n{transcript}",
                    'temperature': 0.7,
                    'max_tokens': 150
                }

                response = requests.post("https://api.openai.com/v1/completions", headers=headers, json=data)
                if response.status_code == 200:
                    response_text = response.json()['choices'][0]['text']
                    if save_path:
                        with open(save_path, 'w', encoding='utf-8') as file:
                            file.write(response_text)
                        print(f"Response saved to {save_path}")
                        self.system_message_label['text'] = "Response received."
                    else:
                        print("No save path provided.")
                else:
                    print(f"Error sending to API: {response.status_code}\n{response.text}")
            else:
                print("Make sure API Key and Model selection are set.")

        elif service == "Local LLM":
            llm_url = self.entry_llm_url.get()
            prompt = self.entry_prompt.get()
            system_message = self.entry_system_message.get()  # Adjust as needed
            save_path = self.entry_save_path.get()

            if llm_url:
                data = {
                    "model": "local-model",  # This field might be unused but is included for completeness
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": f"{prompt}\n\n{transcript}"}
                    ],
                    "temperature": 0.7
                }
                self.system_message_label['text'] = "Sending to API..."
                response = requests.post(llm_url, json=data)
                if response.status_code == 200:
                    self.system_message_label['text'] = "Response received."
                    response_data = response.json()
                    if 'choices' in response_data and len(response_data['choices']) > 0:
                        response_text = response_data['choices'][0]['message']['content']
                        if save_path:
                            with open(save_path, 'w', encoding="utf-8") as file:
                                file.write(response_text)
                            print(f"Response saved to {save_path}")
                        else:
                            print("No save path provided.")
                    else:
                        print("Received unexpected response format:", response_data)
                else:
                    print(f"Error sending to local LLM: {response.status_code}\n{response.text}")
            else:
                print("Make sure the LLM URL is set.")

if __name__ == "__main__":
    root = tk.Tk()
    
    # Set the overall font for the application
    default_font = ('Segoe UI', 10)
    root.option_add("*Font", default_font)
    root.configure(bg='#2D2D2D')


    # Initialize the app with a custom style
    app = TranscriptionApp(root)
    root.mainloop()