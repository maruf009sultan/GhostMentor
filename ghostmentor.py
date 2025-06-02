import os
import json
import asyncio
import numpy as np
import cv2
from PIL import Image, ImageGrab
import google.generativeai as genai
import pygame
import win32gui
import win32con
import logging
from threading import Thread
from queue import Queue
import time
import ctypes
import keyboard
from datetime import datetime
import textwrap
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Gemini API setup
GEMINI_API_KEY = "xxxxxxxxxxxxxxx"
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.0-flash-exp"

# Audio constants
SAMPLING_RATE = 16000
CHUNK_SIZE = 1024

# Command-line arguments
parser = argparse.ArgumentParser(description="Live Support HUD Application")
parser.add_argument('-f', '--full', action='store_true', help="Run in full mode with speech input (default)")
parser.add_argument('-s', '--silent', action='store_true', help="Run in silent mode without speech input (for noisy/outdoor usage)")
args = parser.parse_args()

# Validate arguments
if args.full and args.silent:
    logger.error("Cannot use both -f and -s parameters simultaneously")
    exit(1)

# Conditional imports for speech input
use_speech = not args.silent  # Use speech unless -s is specified
if use_speech:
    import pyaudio
    from faster_whisper import WhisperModel

# Global variables
current_transcript = ""
overlay_text = "Ready..."
screen = None
font = None
text_queue = Queue()
last_response_time = time.time()
hwnd = None
loop = None
scroll_offset = 0  # Track scroll position
conversation_history = []  # Store (question, answer) pairs
audio_buffer = []
buffer_duration = 5  # seconds
buffer_frames = SAMPLING_RATE * buffer_duration if use_speech else 0

def wrap_text(text, width, font):
    """Wrap text to fit within the given pixel width."""
    lines = []
    for paragraph in text.split('\n'):
        wrapped_lines = textwrap.wrap(paragraph, width=50)  # Adjust for 18pt font
        lines.extend(wrapped_lines if wrapped_lines else [''])
    return lines

def capture_screen():
    """Capture the screen and return as a PIL Image."""
    try:
        screenshot = ImageGrab.grab()
        logger.debug("Screen captured successfully")
        return screenshot
    except Exception as e:
        logger.error(f"Screen capture error: {e}")
        return None

def save_screenshot():
    """Save a screenshot to the local directory."""
    try:
        screenshot = capture_screen()
        if screenshot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            screenshot.save(filename)
            logger.info(f"Screenshot saved as {filename}")
            return filename
        else:
            logger.warning("Failed to capture screenshot")
            return None
    except Exception as e:
        logger.error(f"Error saving screenshot: {e}")
        return None

async def send_to_gemini(image, text):
    """Send screen image and transcribed text to Gemini API."""
    global conversation_history
    try:
        if image is None:
            logger.warning("No screen capture available")
            text_queue.put("Error: No screen capture available")
            return None
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction="You are an expert assistant for screen analysis."
        )
        img_array = np.array(image)
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.png', img_rgb)
        img_bytes = buffer.tobytes()

        if text.strip():
            prompt = f"User question: {text}"
        else:
            prompt = ("You are an expert assistant for helping in interviews,solve math,physics,logic,coding ,any problem on screen and show step by step process to solve it/send code and direct solution.")
        logger.info(f"Sending to Gemini: prompt='{prompt}'")
        response = model.generate_content([{"mime_type": "image/png", "data": img_bytes}, prompt], stream=True)

        full_response = ""
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                logger.info(f"Received from Gemini: {chunk.text}")
        if full_response:
            # Store question and answer in history
            conversation_history.append((text, full_response))
            # Format history for display
            history_text = "\n---\n".join(f"Q: {q if q.strip() else '[No question]'}\nA: {a}" for q, a in conversation_history)
            text_queue.put(history_text)
            logger.info(f"Full Gemini response: {full_response}")
        else:
            text_queue.put("No response from Gemini")
            logger.warning("Empty Gemini response")
        return full_response
    except Exception as e:
        logger.error(f"Error sending to Gemini: {e}")
        text_queue.put(f"Gemini error: {str(e)}")
        return None

async def process_gemini():
    """Process transcription and screenshot."""
    global current_transcript
    try:
        image = capture_screen()
        await send_to_gemini(image, current_transcript)
    except Exception as e:
        logger.error(f"Gemini processing error: {e}")
        text_queue.put(f"Processing error: {str(e)}")

def whisper_transcribe():
    """Continuously transcribe audio using Faster-Whisper."""
    global current_transcript, audio_buffer
    try:
        while True:
            try:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                audio_buffer.append(audio_np)

                # Process when buffer reaches 5 seconds
                total_frames = len(audio_buffer) * CHUNK_SIZE
                if total_frames >= buffer_frames:
                    full_audio = np.concatenate(audio_buffer)
                    audio_buffer = []  # Clear buffer
                    logger.info(f"Processing audio with duration {total_frames/SAMPLING_RATE:.3f}s")
                    segments, info = whisper_model.transcribe(full_audio, beam_size=5, language='en')
                    text = " ".join(segment.text for segment in segments).strip()
                    logger.info(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
                    if text:
                        current_transcript = text
                        logger.info(f"Transcribed: {text}")
                    else:
                        logger.debug("No transcription produced")
            except Exception as e:
                logger.error(f"Whisper transcription error: {e}")
    except KeyboardInterrupt:
        logger.info("Whisper transcription stopped")

def update_overlay():
    """Update overlay text from the queue."""
    global overlay_text, last_response_time, scroll_offset
    try:
        while not text_queue.empty():
            overlay_text = text_queue.get()
            # Calculate wrapped lines to set scroll_offset to show latest response
            wrapped_lines = wrap_text(overlay_text, 780, font)
            max_lines = 8  # Maximum lines visible in HUD
            scroll_offset = max(0, len(wrapped_lines) - max_lines)  # Show latest lines
            last_response_time = time.time()
            logger.info(f"Overlay updated with: {overlay_text}, scroll_offset={scroll_offset}")
    except Exception as e:
        logger.error(f"Overlay update error: {e}")

def keep_on_top():
    """Ensure the window stays on top."""
    try:
        if hwnd:
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            logger.debug("Reasserted HWND_TOPMOST")
    except Exception as e:
        logger.error(f"Keep on top error: {e}")

def create_hud():
    """Create a floating HUD window with Pygame."""
    global screen, font, hwnd
    try:
        pygame.init()
        os.environ['SDL_VIDEO_WINDOW_POS'] = "100,100"
        screen = pygame.display.set_mode((800, 200), pygame.NOFRAME | pygame.SRCALPHA)
        pygame.display.set_caption("HUD Overlay")
        logger.info("Pygame window initialized")

        hwnd = pygame.display.get_wm_info()['window']
        ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        ex_style |= win32con.WS_EX_LAYERED | win32con.WS_EX_NOACTIVATE
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, ex_style)

        WDA_EXCLUDEFROMCAPTURE = 0x00000011
        ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)

        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 100, 100, 800, 200, 0)
        logger.info("Window properties set: WS_EX_LAYERED, WS_EX_NOACTIVATE, WDA_EXCLUDEFROMCAPTURE, HWND_TOPMOST")

        font = pygame.font.SysFont('arial', 18)
        screen.set_alpha(179)  # 70% transparency
        logger.info("HUD window created and visible")
    except Exception as e:
        logger.error(f"HUD creation error: {e}")
        raise

def setup_keybindings():
    """Set up universal key bindings using keyboard library."""
    def on_ctrl_h():
        logger.info("Key binding: Ctrl + H pressed")
        save_screenshot()

    def on_ctrl_enter():
        logger.info("Key binding: Ctrl + Enter pressed")
        text_queue.put("Processing...")
        asyncio.run_coroutine_threadsafe(process_gemini(), loop)

    def on_ctrl_g():
        logger.info("Key binding: Ctrl + G pressed")
        global current_transcript, conversation_history
        current_transcript = ""
        conversation_history = []  # Clear history
        text_queue.put("Ready...")

    def on_alt_f4():
        logger.info("Key binding: Alt + F4 pressed")
        global running
        running = False

    try:
        keyboard.add_hotkey('ctrl+h', on_ctrl_h)
        keyboard.add_hotkey('ctrl+enter', on_ctrl_enter)
        keyboard.add_hotkey('ctrl+g', on_ctrl_g)
        keyboard.add_hotkey('alt+f4', on_alt_f4)
        logger.info("Universal key bindings set up")
    except Exception as e:
        logger.error(f"Error setting up key bindings: {e}")

def run_asyncio_loop(loop):
    """Run asyncio event loop in a separate thread."""
    asyncio.set_event_loop(loop)
    loop.run_forever()

def main():
    """Main function to start HUD."""
    global loop, running, scroll_offset, stream, whisper_model, audio

    try:
        # Initialize Faster-Whisper and audio if using speech
        if use_speech:
            global whisper_model, audio, stream
            MODEL_SIZE = "base"
            try:
                whisper_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
                logger.info(f"Faster-Whisper model '{MODEL_SIZE}' loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Faster-Whisper model: {e}")
                exit(1)
            try:
                audio = pyaudio.PyAudio()
                stream = audio.open(format=pyaudio.paInt16, channels=1, rate=SAMPLING_RATE, input=True, frames_per_buffer=CHUNK_SIZE)
                stream.start_stream()
                logger.info("Audio stream started")
            except Exception as e:
                logger.error(f"Audio setup error: {e}")
                exit(1)

            # Start Whisper transcription in a separate thread
            transcription_thread = Thread(target=whisper_transcribe, daemon=True)
            transcription_thread.start()
            logger.info("Whisper transcription started")

        # Create HUD window
        create_hud()

        # Initialize asyncio loop for Gemini
        loop = asyncio.new_event_loop()
        asyncio_thread = Thread(target=run_asyncio_loop, args=(loop,), daemon=True)
        asyncio_thread.start()

        # Set up universal key bindings
        setup_keybindings()

        # Main Pygame loop
        dragging = False
        offset = (0, 0)
        clock = pygame.time.Clock()
        running = True
        while running:
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        logger.info("Window close event detected")
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1:
                            mouse_x, mouse_y = event.pos
                            if 0 <= mouse_x <= 800 and 0 <= mouse_y <= 200:
                                dragging = True
                                offset = (mouse_x, mouse_y)
                                logger.debug("Started dragging HUD")
                    elif event.type == pygame.MOUSEBUTTONUP:
                        if event.button == 1:
                            dragging = False
                            logger.debug("Stopped dragging HUD")
                    elif event.type == pygame.MOUSEMOTION and dragging:
                        mouse_x, mouse_y = event.pos
                        current_x, current_y = map(int, os.environ.get('SDL_VIDEO_WINDOW_POS', '100,100').split(','))
                        new_x = current_x + (mouse_x - offset[0])
                        new_y = current_y + (mouse_y - offset[1])
                        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{new_x},{new_y}"
                        pygame.display.set_mode((800, 200), pygame.NOFRAME | pygame.SRCALPHA)
                        logger.debug(f"Dragged HUD to ({new_x}, {new_y})")
                    elif event.type == pygame.MOUSEWHEEL:
                        mouse_x, mouse_y = pygame.mouse.get_pos()
                        if 0 <= mouse_x <= 800 and 0 <= mouse_y <= 200:
                            scroll_offset -= event.y  # Scroll up: +1, down: -1
                            wrapped_lines = wrap_text(overlay_text, 780, font)
                            scroll_offset = max(0, min(scroll_offset, len(wrapped_lines) - 8))  # Bound scroll
                            logger.debug(f"Scrolled HUD, offset={scroll_offset}")

                # Update overlay text
                update_overlay()
                keep_on_top()

                # Render HUD with wrapped text and scroll
                screen.fill((0, 0, 0, 179))  # 70% transparency
                wrapped_lines = wrap_text(overlay_text, 780, font)
                max_lines = 8  # Fit 8 lines in 200px height
                visible_lines = wrapped_lines[scroll_offset:scroll_offset + max_lines]
                for i, line in enumerate(visible_lines):
                    text_surface = font.render(line, True, (255, 255, 255))
                    screen.blit(text_surface, (10, 10 + i * 22))

                pygame.display.flip()
                clock.tick(60)
            except Exception as e:
                logger.error(f"Pygame loop error: {e}")

    except Exception as e:
        logger.error(f"Main error: {e}")
        raise
    finally:
        if use_speech:
            stream.stop_stream()
            stream.close()
            audio.terminate()
            logger.info("Audio resources cleaned up")
        pygame.quit()
        logger.info("Pygame window closed")
        keyboard.unhook_all()
        logger.info("Keyboard bindings removed")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
