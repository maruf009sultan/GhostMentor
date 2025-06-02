# 👻 GhostMentor: The Phantom Code

---

### *Unseen. Unheard. Unstoppable.*

**Your shadow ally in the digital underworld.**

An open-source solution inspired by [fuckleetcode](https://www.interviewcoder.co/), designed to give you the ultimate edge in coding battles without compromise.

---

GhostMentor isn’t just code—it’s a **rebellion**.
A real-time HUD (Heads-Up Display) that **whispers solutions, cracks problems, and gives you the edge** in coding battles, interviews, and contests.
Built with **Gemini API, Faster-Whisper, OpenCV, and Pygame**, it’s the ultimate tool for those who thrive in the shadows.

---

⚠️ **Disclaimer:**
*GhostMentor is a proof-of-concept for educational exploration.*
Use it to push boundaries, but respect the rules.
**We’re not liable if you go rogue.**

---
## 👻 GhostMentor in Action

Watch the phantom code in motion — proof of GhostMentor’s stealth and power:

https://github.com/maruf009sultan/GhostMentor/blob/main/VID-20250602-WA0007.mp4



---

## 🔥 Why Choose the Shadow?

GhostMentor is your invisible partner, designed for those who **dare to dominate**.
It’s not about playing fair—it’s about playing smart.
With stealth tech and AI wizardry, it delivers:

| **Feature**            | **Why It’s Epic**                                                  |
| ---------------------- | ------------------------------------------------------------------ |
| **Stealth Mode**       | Bypasses screen-sharing detection like a digital ghost.            |
| **Real-Time Analysis** | Processes screens and speech faster than a proctor can blink.      |
| **Dual Modes**         | Full mode (-f) for voice-driven power, silent mode (-s) for secret. |
| **Elite Tech Stack**   | Gemini, Faster-Whisper, OpenCV, Pygame—pure cyberpunk magic.       |

---

---

# 🕷️ GhostMentor_Unethical.py — The Vanishing Act Module

**Stealth beyond stealth.** For those who don’t want to be seen, even by the system itself.

---

## 🛠️ Ultra-Stealth Features

* **🪞  No taskbar. No tabs. No traces.**
* **🧟‍♂️Room for Mimic a system process** — rename to `NVIDIA_Service.exe`, `winlogon64.exe`, etc.
* **🧰 Ready-to-build EXE via PyInstaller:**
    ```bash
        pyinstaller --noconfirm --onefile --windowed --name "NVIDIA_Service" --icon "path/to/your/icon.ico" ghostmentor_unethical.py
    ```
* **🎭 Task Manager deception:**
    * Custom process name + fake icon
    * Registry startup persistence (`HKLM\...\Run`)
    * Optional fake publisher or crash logs

---

## ⚔️ Why It Outclasses 99.9% “Stealth” Tools

| Capability                         | GhostMentor_Unethical | Most Tools   |
| :--------------------------------- | :-------------------- | :----------- |
| No window/taskbar presence         | ✅                    | ❌           |
| Disguisable process name/icon      | ✅                    | ❌           |
| Registry + autorun support         | ✅                    | ⚠️ Manual    |
| Zoom/Meet/screen recorder bypass   | ✅                    | ❌           |
| Fully customizable .exe build      | ✅                    | ❌           |
| Open-source                        | ✅                    | ❌           |

---

**Use it like a ghost. Build it like a weapon.**


## ✅ Use Cases: Rule the Underworld

GhostMentor is for the **bold**, the **cunning**, the coders who **live on the edge**:

* 👨‍💻 **Crush Interviews:** Ace coding interviews with answers whispered in real-time, invisible to screen-sharing apps.
* 👾 **Contest Phantom:** Dominate LeetCode or HackerRank with a silent mentor guiding every move.
* 🧠 **Invisible Genius:** Get on-screen solutions for code, math, or UI—undetected, always.
* 🔕 **Noisy or Silent:** From chaotic labs to proctored exams, GhostMentor adapts and delivers.

---

## 🛠️ Features: Your Arsenal of Shadows

* **Real-Time Screen Analysis:** Gemini API dissects your screen for instant insights.
* **Speech Transcription:** Faster-Whisper captures your voice, even in noisy environments (full mode only).
* **Transparent HUD:** Scrollable, always on top, never seen.
* **Key Bindings:** Ctrl+H (screenshot), Ctrl+Enter (analyze), Ctrl+G (reset), Alt+F4 (vanish).
* **Anti-Detection:** Excluded from screen captures, invisible to Zoom, Teams, or proctors.

---
## ⚠️ Shadow Bypass: Invisible Among the Watchers

GhostMentor has **successfully ghosted** through the following screen-sharing and recording defenses:

- 🕵️‍♂️ **Zoom Screen Share** — Undetectable, invisible.
- 👁️‍🗨️ **Google Meet Screen Share** — Phantom mode engaged.
- 🎥 **ShareX Screen Recorder** — Silent as the grave.

…and many more.

> **Theoretical Mastery:**  
> Every modern web or browser-based screen-sharing tool? GhostMentor can **bypass** them — because shadows don’t leave footprints.

Embrace invisibility. Outrun the watchful eyes. Stay **unseen** and **unstoppable**. ( *no extra settings needed , unconditional.)

## 🎯 Hack the Master Prompt: Dark Uses for GhostMentor

GhostMentor isn’t just a tool — it’s a weapon. Customize your master prompt to unleash your inner shadow operator. Here’s how to twist the rules and dominate:

| Role                          | What You Can Do (No Rules, Just Power)                              |
|-------------------------------|--------------------------------------------------------------------|
| 🕵️‍♂️ **The Phantom Teacher**       | Make AI tutor to help you with math, physics, anything. |
| 🎭 **The Interview Ghost**           | Get whispered cheat-sheet style hints *live* during interviews — unseen by all but you.          |
| ⚔️ **The Contest Phantom**            | Use AI to silently decode puzzles and reverse-engineer solutions during competitions.              |
| 🧟 **The Code Whisperer**             | Get AI to debug your code. |
| 🔮 **The Market Manipulator**         | Deploy AI to analyze and manipulate data patterns in real-time for unfair trading edges.           |
> *This is the shadow world — play hard, play smart, but beware: power corrupts. GhostMentor doesn’t judge, it obeys.*

---

**Warning:** These are ideas for those who dare to push boundaries. Use responsibly. Or don’t.



## 🧑‍💻 Technical Deep Dive: The Dark Arts

GhostMentor is a **masterpiece of code and cunning**, built for AI geeks who love the grind.
Here’s the tech that powers the shadow:

### ✅ Gemini Image + Prompt Flow

* **Input:** Screen captured via `PIL.ImageGrab.grab()`, converted and encoded as PNG bytes for API magic.
* **Prompt:** Dynamic — from whispered questions to default deep code analysis.
* **Output:** Real-time streaming answers fuel your HUD with instant insights.

```python
img_array = np.array(image)
img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
_, buffer = cv2.imencode(".png", img_rgb)
img_bytes = buffer.tobytes()
response = model.generate_content(
    [{"mime_type": "image/png", "data": img_bytes}, prompt], stream=True)
```

*Why It’s Badass:*
Fuses vision and NLP to solve coding problems or analyze UI in milliseconds.

---

### ✅ Speech Buffering

* **Model:** Faster-Whisper (base, int8 precision, CPU) for low-latency transcription.
* **Pipeline:** 16kHz audio chunks, buffered 5 seconds, transcribed with beam search.

```python
audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
audio_buffer.append(audio_np)
if len(audio_buffer) * CHUNK_SIZE >= SAMPLING_RATE * BUFFER_DURATION:
    full_audio = np.concatenate(audio_buffer)
    segments, info = whisper_model.transcribe(full_audio, beam_size=5, language="en")
```

*Why It’s Badass:*
Cuts through noise to deliver crystal-clear commands in real-time.

---

### ✅ Screenshot Byte Compression

* Efficient PNG encoding shrinks data size for lightning-fast Gemini calls.

```python
img_array = np.array(image)
img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
_, buffer = cv2.imencode(".png", img_rgb)
img_bytes = buffer.tobytes()
```

*Why It’s Badass:*
No lag, no fuzz, just raw speed.

---

### ✅ Anti-Overlap Logic

* Keeps HUD on top while making it invisible to screen sharing apps and proctors.

```python
WDA_EXCLUDEFROMCAPTURE = 0x00000011
ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 100, 100, 800, 200, 0)
```

*Why It’s Badass:*
GhostMentor stays in the shadows—undetectable and untouchable.

---

### ✅ Speech Transcription Safety Logic

* Language filtering and beam search keep transcription clean and accurate.

```python
segments, info = whisper_model.transcribe(full_audio, beam_size=5, language="en")
if info.language_probability > 0.9:
    text = " ".join(segment.text for segment in segments).strip()
```

*Why It’s Badass:*
Reliable, even in chaotic battlefields.

---

### ✅ Async Gemini Handling

* Runs queries asynchronously for fluid, instant HUD updates.

```python
loop = asyncio.new_event_loop()
asyncio_thread = Thread(target=run_asyncio_loop, args=(loop,), daemon=True)
asyncio_thread.start()
response = model.generate_content(
    [{"mime_type": "image/png", "data": img_bytes}, prompt], stream=True)
```

*Why It’s Badass:*
No lag, no freeze, just smooth operator performance.

---

## 📦 Installation: Summon the Phantom

```bash
git clone https://github.com/maruf009sultan/GhostMentor.git
cd GhostMentor
pip install numpy opencv-python pillow google-generativeai pygame pyaudio faster-whisper keyboard pywin32
```

1. Grab your Gemini API key from [AI studio](https://aistudio.google.com/)
2. Update `API_KEY` in `ghostmentor.py` (it runs on local machine so I didn't used .env):

   ```python
   API_KEY = "your-api-key-here"
   ```
3. Run your ghost:

* **Full mode (voice-enabled):**
  `python ghostmentor.py -f`

* **Silent mode (stealth only):**
  `python ghostmentor.py -s`

---

## 🔑 Key Bindings: Control the Darkness

| Key        | Action                          |
| ---------- | ------------------------------- |
| Ctrl+H     | Save screenshot—flex your rig.  |
| Ctrl+Enter | Trigger Gemini analysis now.    |
| Ctrl+G     | Clear transcript/history reset. |
| Alt+F4     | Vanish GhostMentor instantly.   |

---

## 📜 GhostMentor Shadow License (GSL)

GhostMentor isn’t your average open-source toy.
It’s locked under the **GhostMentor Shadow License (GSL):**

See `LICENSE.md` for the full spell.

---

⚠️ **Disclaimer: Stay in the Shadows**
GhostMentor pushes AI limits—explore, don’t destroy.
Developers aren’t liable for misuse. Stay sharp, stay invisible.

---

## 🌑 Join the Underground

GhostMentor is for the **renegades**, the coders who **thrive in the dark**.
Star ⭐ the repo, share with fellow shadow coders, and watch for updates.
Got a killer idea? Open an issue or PR—only the bold survive.

---

> **"The code doesn’t lie. Neither does GhostMentor."**

---

## 🖤 Built by Rebels, for Rebels

Crafted with tools that scream power:

* **Gemini API:** Vision and reasoning that see your screen’s soul.
* **Faster-Whisper:** Hears through the noise, even in chaos.
* **OpenCV & PIL:** Pixel-perfect screen analysis.
* **Pygame:** A HUD straight out of a cyberpunk fever dream.
* **Win32 API:** Stealth tricks to keep you under the radar.

---

**Clone it. Run it. Dominate.**
