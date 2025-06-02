
# === Imports ===
import argparse
import asyncio
import ctypes
import logging
import os
from threading import Thread
import textwrap
import time
from datetime import datetime
from queue import Queue
from typing import List, Optional, Tuple

import cv2
import google.generativeai as genai
import numpy as np
import pygame
import win32con
import win32gui
from PIL import Image, ImageGrab
import keyboard

# === Logging Configuration ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# === Constants ===
# Gemini API
GEMINI_API_KEY = ""
GEMINI_MODEL = "gemini-2.0-flash-exp"

# HUD Display
HUD_WIDTH = 500
HUD_HEIGHT = 200
HUD_TRANSPARENCY = 179  # 70% transparency (0-255)
MAX_VISIBLE_LINES = 8   # Lines visible in HUD

# === Global Variables ===
current_transcript: str = ""                    # No speech input, always empty
overlay_text: str = "Ready..."                 # HUD display text
screen: Optional[pygame.Surface] = None        # Pygame display surface
font: Optional[pygame.font.Font] = None        # Font for HUD text
text_queue: Queue = Queue()                    # Queue for HUD updates
last_response_time: float = time.time()        # Last response timestamp
hwnd: Optional[int] = None                     # Windows handle for HUD
loop: Optional[asyncio.AbstractEventLoop] = None  # Asyncio event loop
scroll_offset: int = 0                         # Scroll position for history
conversation_history: List[Tuple[str, str]] = []  # (question, answer) pairs

# === Command-Line Arguments ===
def parse_args() -> None:
    """Parse command-line arguments (silent mode only)."""
    parser = argparse.ArgumentParser(description="Live Support HUD Application (Silent Mode)")
    parser.add_argument(
        "-s", "--silent", action="store_true",
        help="Run in silent mode without speech input (default, only mode available)"
    )
    args = parser.parse_args()
    logger.info("Running in silent mode (no speech input)")
    return args

# === Text Rendering ===
def wrap_text(text: str, width: int, font: pygame.font.Font) -> List[str]:
    """Wrap text to fit within the given pixel width.

    Args:
        text: Input text to wrap.
        width: Maximum width in pixels.
        font: Pygame font for rendering.

    Returns:
        List of wrapped text lines.
    """
    lines = []
    for paragraph in text.split("\n"):
        wrapped_lines = textwrap.wrap(paragraph, width=50)  # Adjusted for 18pt font
        lines.extend(wrapped_lines if wrapped_lines else [""])
    return lines

# === Screen Capture ===
def capture_screen() -> Optional[Image.Image]:
    """Capture the screen and return as a PIL Image.

    Returns:
        PIL Image or None if capture fails.
    """
    try:
        screenshot = ImageGrab.grab()
        logger.debug("Screen captured successfully")
        return screenshot
    except Exception as e:
        logger.error(f"Screen capture error: {e}")
        return None

def save_screenshot() -> Optional[str]:
    """Save a screenshot to the local directory.

    Returns:
        Filename of saved screenshot or None if failed.
    """
    try:
        screenshot = capture_screen()
        if screenshot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            screenshot.save(filename)
            logger.info(f"Screenshot saved as {filename}")
            return filename
        logger.warning("Failed to capture screenshot")
        return None
    except Exception as e:
        logger.error(f"Error saving screenshot: {e}")
        return None

# === Gemini API Interaction ===
async def send_to_gemini(image: Optional[Image.Image], text: str) -> Optional[str]:
    """Send screen image and transcribed text to Gemini API.

    Args:
        image: PIL Image of the screen.
        text: Transcribed text (always empty in silent mode).

    Returns:
        Gemini response text or None if failed.
    """
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
        _, buffer = cv2.imencode(".png", img_rgb)
        img_bytes = buffer.tobytes()

        prompt = (
            f"User question: {text}" if text.strip() else
            "You are an expert assistant for helping in interviews, solve math, physics, logic, coding, "
            "any problem on screen and show step by step process to solve it/send code and direct solution."
        )

        logger.info(f"Sending to Gemini: prompt='{prompt[:100]}...'")
        response = model.generate_content([{"mime_type": "image/png", "data": img_bytes}, prompt], stream=True)

        full_response = ""
        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                logger.info(f"Received from Gemini: {chunk.text[:100]}...")
        if full_response:
            conversation_history.append((text, full_response))
            history_text = "\n---\n".join(
                f"Q: {q if q.strip() else '[No question]'}\nA: {a}" for q, a in conversation_history
            )
            text_queue.put(history_text)
            logger.info(f"Full Gemini response: {full_response[:100]}...")
            return full_response
        else:
            text_queue.put("No response from Gemini")
            logger.warning("Empty Gemini response")
            return None
    except Exception as e:
        logger.error(f"Error sending to Gemini: {e}")
        text_queue.put(f"Gemini error: {str(e)}")
        return None

async def process_gemini() -> None:
    """Process screenshot for Gemini analysis."""
    try:
        image = capture_screen()
        await send_to_gemini(image, current_transcript)
    except Exception as e:
        logger.error(f"Gemini processing error: {e}")
        text_queue.put(f"Processing error: {str(e)}")

# === HUD Management ===
def update_overlay() -> None:
    """Update overlay text from the queue."""
    global overlay_text, last_response_time, scroll_offset
    try:
        while not text_queue.empty():
            overlay_text = text_queue.get()
            wrapped_lines = wrap_text(overlay_text, HUD_WIDTH - 20, font)
            scroll_offset = max(0, len(wrapped_lines) - MAX_VISIBLE_LINES)
            last_response_time = time.time()
            logger.info(f"Overlay updated, scroll_offset={scroll_offset}")
    except Exception as e:
        logger.error(f"Overlay update error: {e}")

def keep_on_top() -> None:
    """Ensure the window stays on top."""
    try:
        if hwnd:
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
            logger.debug("Reasserted HWND_TOPMOST")
    except Exception as e:
        logger.error(f"Keep on top error: {e}")

def create_hud() -> None:
    """Create a floating HUD window with Pygame, hidden from taskbar and Alt+Tab."""
    global screen, font, hwnd
    try:
        # Hide console window during script execution
        console_hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if console_hwnd:
            ctypes.windll.user32.ShowWindow(console_hwnd, 0)  # SW_HIDE
            logger.info("Console window hidden")

        pygame.init()
        os.environ["SDL_VIDEO_WINDOW_POS"] = "100,100"
        screen = pygame.display.set_mode((HUD_WIDTH, HUD_HEIGHT), pygame.NOFRAME | pygame.SRCALPHA)
        pygame.display.set_caption("HUD Overlay")
        logger.info("Pygame window initialized")

        hwnd = pygame.display.get_wm_info()["window"]
        ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        ex_style |= win32con.WS_EX_LAYERED | win32con.WS_EX_NOACTIVATE | win32con.WS_EX_TOOLWINDOW
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, ex_style)

        WDA_EXCLUDEFROMCAPTURE = 0x00000011
        ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 100, 100, HUD_WIDTH, HUD_HEIGHT, 0)
        logger.info("Window properties set: layered, no-activate, toolwindow, excluded from capture")

        font = pygame.font.SysFont("arial", 18)
        screen.set_alpha(HUD_TRANSPARENCY)
        logger.info("HUD window created and visible")
    except Exception as e:
        logger.error(f"HUD creation error: {e}")
        raise

# === Key Bindings ===
def setup_keybindings() -> None:
    """Set up universal key bindings using keyboard library."""
    def on_ctrl_h():
        logger.info("Ctrl+H: Saving screenshot")
        save_screenshot()

    def on_ctrl_enter():
        logger.info("Ctrl+Enter: Sending to Gemini")
        text_queue.put("Processing...")
        asyncio.run_coroutine_threadsafe(process_gemini(), loop)

    def on_ctrl_g():
        logger.info("Ctrl+G: Resetting history")
        global current_transcript, conversation_history
        current_transcript = ""
        conversation_history = []
        text_queue.put("Ready...")

    def on_alt_f4():
        logger.info("Alt+F4: Exiting")
        global running
        running = False

    try:
        keyboard.add_hotkey("ctrl+h", on_ctrl_h)
        keyboard.add_hotkey("ctrl+enter", on_ctrl_enter)
        keyboard.add_hotkey("ctrl+g", on_ctrl_g)
        keyboard.add_hotkey("alt+f4", on_alt_f4)
        logger.info("Universal key bindings set up")
    except Exception as e:
        logger.error(f"Error setting up key bindings: {e}")

# === Asyncio Support ===
def run_asyncio_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Run asyncio event loop in a separate thread."""
    asyncio.set_event_loop(loop)
    loop.run_forever()

# === Main Application ===
def main() -> None:
    """Main function to start HUD."""
    global loop, running, scroll_offset

    parse_args()  # Silent mode only

    try:
        # Configure Gemini API
        genai.configure(api_key=GEMINI_API_KEY)

        # Create HUD window
        create_hud()

        # Initialize asyncio loop for Gemini
        loop = asyncio.new_event_loop()
        Thread(target=run_asyncio_loop, args=(loop,), daemon=True).start()

        # Set up universal key bindings
        setup_keybindings()

        # Main Pygame loop
        clock = pygame.time.Clock()
        running = True
        while running:
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        logger.info("Window close event detected")
                    elif event.type == pygame.MOUSEWHEEL:
                        mouse_x, mouse_y = pygame.mouse.get_pos()
                        if 0 <= mouse_x <= HUD_WIDTH and 0 <= mouse_y <= HUD_HEIGHT:
                            scroll_offset -= event.y  # Scroll up: +1, down: -1
                            wrapped_lines = wrap_text(overlay_text, HUD_WIDTH - 20, font)
                            scroll_offset = max(0, min(scroll_offset, len(wrapped_lines) - MAX_VISIBLE_LINES))
                            logger.debug(f"Scrolled HUD, offset={scroll_offset}")

                # Update overlay text
                update_overlay()
                keep_on_top()

                # Render HUD with wrapped text and scroll
                screen.fill((0, 0, 0, HUD_TRANSPARENCY))
                wrapped_lines = wrap_text(overlay_text, HUD_WIDTH - 20, font)
                visible_lines = wrapped_lines[scroll_offset:scroll_offset + MAX_VISIBLE_LINES]
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
        pygame.quit()
        logger.info("Pygame window closed")
        keyboard.unhook_all()
        logger.info("Keyboard bindings removed")

if __name__ == "__main__":
    try:
        logger.info("Starting StudyHUD")
        main()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
