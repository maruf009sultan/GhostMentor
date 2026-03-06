<div align="center">

# 👻 GhostMentor

### *The AI-Powered Accessibility Assistant*

**Unseen. Unheard. Unstoppable.**

*Your shadow ally in digital accessibility and AI research.*

---

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Gemini API](https://img.shields.io/badge/Gemini%20API-Integrated-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/License-GSL-purple?style=for-the-badge)](./LICENSE.md)
[![Open Source](https://img.shields.io/badge/Open%20Source-Yes-green?style=for-the-badge&logo=github&logoColor=white)](https://github.com/maruf009sultan/GhostMentor)
[![Research](https://img.shields.io/badge/Purpose-Research%20%7C%20Educational-orange?style=for-the-badge)]()

---

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active%20Development-brightgreen?style=flat-square" alt="Status"/>
  <img src="https://img.shields.io/badge/Platform-Windows-informational?style=flat-square&logo=windows&logoColor=white" alt="Platform"/>
  <img src="https://img.shields.io/badge/AI-Gemini%20%7C%20Whisper-9cf?style=flat-square" alt="AI"/>
</p>

</div>

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Features](#-features)
- [🎓 Academic & Research Applications](#-academic--research-applications)
- [🔬 Technical Architecture](#-technical-architecture)
- [📦 Installation](#-installation)
- [🚀 Usage Guide](#-usage-guide)
- [⌨️ Keyboard Shortcuts](#️-keyboard-shortcuts)
- [🛡️ Security & Privacy Considerations](#️-security--privacy-considerations)
- [📊 System Requirements](#-system-requirements)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

**GhostMentor** is an innovative AI-powered accessibility and research tool designed to assist users in educational environments, accessibility testing, and assistive technology development. Built with cutting-edge technologies including **Google Gemini API**, **Faster-Whisper**, **OpenCV**, and **Pygame**, GhostMentor represents a significant advancement in real-time AI assistance systems.

### 🎯 Core Purpose

GhostMentor serves multiple legitimate purposes in the academic and research community:

| Purpose | Description |
|---------|-------------|
| 🧪 **Accessibility Research** | Investigating screen capture exclusion technologies for assistive applications |
| 🎓 **Educational Technology** | Developing AI-powered tutoring and learning assistance systems |
| 🔬 **Human-Computer Interaction** | Researching novel HUD interfaces and overlay technologies |
| ♿ **Assistive Technology** | Creating tools for users with visual or cognitive impairments |
| 📊 **AI Research** | Exploring multimodal AI integration (vision + speech + text) |

> ⚠️ **Important Notice**: This tool is intended **exclusively** for educational, research, accessibility testing, and ethical development purposes. Users are responsible for ensuring compliance with all applicable laws, regulations, and institutional policies.

---

## ✨ Features

### 🖥️ Real-Time Screen Analysis

GhostMentor leverages the powerful **Google Gemini API** to analyze screen content in real-time, providing instant insights and assistance:

- **Intelligent Image Recognition**: Captures and processes screen content using advanced computer vision
- **Context-Aware Responses**: AI understands the context of your work and provides relevant assistance
- **Streaming Output**: Real-time response streaming for immediate feedback

### 🎤 Advanced Speech Recognition

Powered by **Faster-Whisper** for accurate and efficient speech transcription:

- **Low-Latency Processing**: Optimized for real-time transcription with minimal delay
- **Multi-Language Support**: Supports multiple languages with high accuracy
- **Noise Resilience**: Advanced algorithms handle noisy environments effectively
- **Beam Search Decoding**: Ensures accurate transcription even with complex speech patterns

### 🎨 Transparent HUD Interface

A modern, non-intrusive interface that integrates seamlessly with your workflow:

- **Always-On-Top Design**: Stays visible without disrupting your primary tasks
- **Scrollable Content**: Review extensive AI responses with ease
- **Minimal Visual Footprint**: Designed to be helpful without being distracting
- **Customizable Appearance**: Adapt the interface to your preferences

### 🔒 Privacy-Preserving Technology

GhostMentor incorporates privacy-focused design principles:

- **Local Processing**: Audio and image processing happens on your machine
- **No Data Storage**: Transient processing without permanent data retention
- **API Security**: Secure communication with AI services via encrypted connections

---

## 🎓 Academic & Research Applications

### 📚 Educational Technology Research

GhostMentor provides an excellent platform for researching:

| Research Area | Application |
|---------------|-------------|
| **Adaptive Learning Systems** | Study how AI assistance affects learning outcomes |
| **Real-Time Tutoring** | Develop and test AI-powered tutoring methodologies |
| **Cognitive Load Theory** | Research optimal information presentation in HUDs |
| **Multimodal Interaction** | Explore combining voice, vision, and text inputs |

### ♿ Accessibility Testing & Development

Essential tool for accessibility researchers and developers:

- **Screen Reader Compatibility Testing**: Test how applications behave with various assistive technologies
- **Visual Impairment Simulation**: Understand user experiences with limited visual access
- **Assistive Technology Development**: Create tools for users with disabilities
- **WCAG Compliance Research**: Investigate accessibility standard implementation

### 🔬 Human-Computer Interaction Studies

Perfect for HCI researchers investigating:

- **Attention Management**: How overlay interfaces affect user focus
- **Information Density**: Optimal information presentation in limited screen space
- **Multimodal Interfaces**: Combining visual and auditory feedback channels
- **Context-Aware Computing**: Systems that adapt to user context and needs

### 🧪 Software Testing & QA

Valuable for software testing professionals:

- **UI/UX Testing**: Automated interface analysis and feedback
- **Accessibility Auditing**: Identify accessibility issues in applications
- **Cross-Platform Compatibility**: Test behavior across different environments
- **Documentation Generation**: Automated documentation from visual analysis

---

## 🔬 Technical Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      GhostMentor Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Screen     │    │    Audio     │    │   Keyboard   │      │
│  │   Capture    │    │   Capture    │    │   Handler    │      │
│  │  (OpenCV)    │    │  (PyAudio)   │    │  (Keyboard)  │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Processing Core                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │   │
│  │  │   Image     │  │   Speech    │  │   Event     │     │   │
│  │  │  Encoding   │  │ Transcribe  │  │  Handling   │     │   │
│  │  │   (PNG)     │  │ (Whisper)   │  │  (Async)    │     │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │   │
│  │         │                │                │            │   │
│  │         └────────────────┼────────────────┘            │   │
│  │                          ▼                             │   │
│  │              ┌─────────────────────┐                   │   │
│  │              │    Prompt Builder   │                   │   │
│  │              └──────────┬──────────┘                   │   │
│  └─────────────────────────┼───────────────────────────────┘   │
│                            │                                   │
│                            ▼                                   │
│              ┌─────────────────────────┐                       │
│              │     Gemini API Call     │                       │
│              │   (Streaming Response)  │                       │
│              └──────────┬──────────────┘                       │
│                         │                                      │
│                         ▼                                      │
│              ┌─────────────────────────┐                       │
│              │     HUD Display         │                       │
│              │    (Pygame Window)      │                       │
│              │   Transparent Overlay   │                       │
│              └─────────────────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 🔧 Core Technologies

#### Gemini API Integration

GhostMentor uses Google's Gemini API for multimodal AI processing:

```python
import google.generativeai as genai

# Image processing pipeline
img_array = np.array(image)
img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
_, buffer = cv2.imencode(".png", img_rgb)
img_bytes = buffer.tobytes()

# API call with streaming
response = model.generate_content(
    [{"mime_type": "image/png", "data": img_bytes}, prompt], 
    stream=True
)
```

#### Faster-Whisper Speech Recognition

Efficient speech-to-text processing:

```python
from faster_whisper import WhisperModel

# Initialize model with optimizations
whisper_model = WhisperModel(
    "base", 
    device="cpu", 
    compute_type="int8"
)

# Audio processing pipeline
audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
segments, info = whisper_model.transcribe(
    full_audio, 
    beam_size=5, 
    language="en"
)
```

#### Screen Capture Technology

Efficient screen capture with OpenCV and PIL:

```python
from PIL import ImageGrab
import cv2
import numpy as np

# Optimized capture
image = ImageGrab.grab()
img_array = np.array(image)
img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
```

#### Window Display Affinity

For privacy-preserving display technology research:

```python
import ctypes
import win32gui
import win32con

# Window display configuration
WDA_EXCLUDEFROMCAPTURE = 0x00000011
ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 100, 100, 800, 200, 0)
```

---

## 📦 Installation

### Prerequisites

- **Python 3.8+** (Python 3.10 recommended)
- **Windows Operating System** (primary platform)
- **Google Gemini API Key** ([Get it here](https://aistudio.google.com/app/apikey))

### 🚀 Quick Start

1. **Clone the Repository**

```bash
git clone https://github.com/maruf009sultan/GhostMentor.git
cd GhostMentor
```

2. **Create Virtual Environment** (Recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy opencv-python pillow google-generativeai pygame pyaudio faster-whisper keyboard pywin32
```

4. **Configure API Key**

Open `ghostmentor.py` and add your Gemini API key:

```python
API_KEY = "your-gemini-api-key-here"
```

> 🔐 **Security Note**: For production use, consider using environment variables or a secure configuration file instead of hardcoding your API key.

5. **Run GhostMentor**

```bash
# Full mode (voice-enabled)
python ghostmentor.py -f

# Silent mode (text-only)
python ghostmentor.py -s
```

### 📁 Project Structure

```
GhostMentor/
├── 📄 ghostmentor.py          # Main application (voice-enabled)
├── 📄 gm_unethical.py         # Stealth research module
├── 📄 requirements.txt        # Python dependencies
├── 📄 LICENSE.md             # GhostMentor Shadow License
├── 📄 README.md              # This documentation
└── 🎬 GhostMentor original.mp4  # Demo video
```

---

## 🚀 Usage Guide

### Operating Modes

GhostMentor offers two primary operating modes:

#### 🔊 Full Mode (`-f`)

Voice-enabled operation with complete functionality:

```bash
python ghostmentor.py -f
```

**Features:**
- ✅ Screen capture and analysis
- ✅ Voice input processing
- ✅ Real-time HUD display
- ✅ Speech transcription

**Best For:**
- Accessibility research
- Multimodal interaction studies
- Voice-controlled assistance development

#### 🔇 Silent Mode (`-s`)

Text-only operation for focused analysis:

```bash
python ghostmentor.py -s
```

**Features:**
- ✅ Screen capture and analysis
- ✅ Real-time HUD display
- ❌ Voice input disabled

**Best For:**
- Quiet environments
- Text-based research
- Minimal system resource usage

### 🎯 Common Use Cases

#### 1. Educational Assistance

Use GhostMentor as a study companion:

```
1. Open your study materials (PDFs, websites, etc.)
2. Launch GhostMentor
3. Press Ctrl+H to capture the screen
4. Press Ctrl+Enter to get AI assistance
```

#### 2. Accessibility Testing

Test application accessibility features:

```
1. Launch the target application
2. Run GhostMentor alongside it
3. Analyze how information is presented
4. Document accessibility improvements
```

#### 3. Research & Development

Integrate into your research workflow:

```
1. Configure your research parameters
2. Use GhostMentor for data collection
3. Analyze AI responses for patterns
4. Document findings for publication
```

---

## ⌨️ Keyboard Shortcuts

| Shortcut | Action | Description |
|----------|--------|-------------|
| `Ctrl + H` | 📸 Screenshot | Capture current screen for analysis |
| `Ctrl + Enter` | ⚡ Analyze | Send captured content to Gemini API |
| `Ctrl + G` | 🔄 Reset | Clear transcript and reset history |
| `Alt + F4` | 🚪 Exit | Close GhostMentor immediately |

---

## 🛡️ Security & Privacy Considerations

### 🔐 Data Handling

GhostMentor is designed with privacy in mind:

| Aspect | Implementation |
|--------|----------------|
| **Local Processing** | Screen and audio processing occurs locally |
| **No Persistent Storage** | Data is not stored permanently |
| **API Security** | Encrypted communication with Gemini API |
| **User Control** | User initiates all captures and analysis |

### ⚠️ Responsible Use Guidelines

**DO:**
- ✅ Use for legitimate educational purposes
- ✅ Employ in accessibility research
- ✅ Utilize for software testing and development
- ✅ Follow institutional and organizational policies
- ✅ Obtain proper permissions when required

**DON'T:**
- ❌ Use to violate terms of service of any platform
- ❌ Employ for academic dishonesty or cheating
- ❌ Use to bypass security measures unauthorized
- ❌ Violate privacy rights of others
- ❌ Engage in any illegal activities

### 🏛️ Institutional Compliance

When using GhostMentor in academic or institutional settings:

1. **Review Policies**: Check your institution's policies on AI assistance tools
2. **Obtain Approval**: Secure necessary approvals from ethics committees
3. **Document Usage**: Maintain records of research usage
4. **Follow Guidelines**: Adhere to field-specific ethical guidelines

---

## 📊 System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| **OS** | Windows 10/11 |
| **Python** | 3.8 or higher |
| **RAM** | 4 GB minimum |
| **Storage** | 500 MB free space |
| **Network** | Internet connection for API |

### Recommended Specifications

| Component | Recommendation |
|-----------|----------------|
| **OS** | Windows 11 |
| **Python** | 3.10+ |
| **RAM** | 8 GB or more |
| **CPU** | Multi-core processor |
| **Network** | Stable broadband connection |

### Dependency Versions

```
numpy>=1.21.0
opencv-python>=4.5.0
pillow>=8.0.0
google-generativeai>=0.1.0
pygame>=2.0.0
pyaudio>=0.2.11
faster-whisper>=0.9.0
keyboard>=0.13.0
pywin32>=300
```

---

## 🤝 Contributing

We welcome contributions from the research and development community!

### How to Contribute

1. **Fork the Repository**
   ```bash
   git fork https://github.com/maruf009sultan/GhostMentor.git
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Follow Python best practices
   - Add appropriate documentation
   - Include tests where applicable

4. **Submit a Pull Request**
   - Describe your changes clearly
   - Reference any related issues
   - Ensure CI passes

### Contribution Guidelines

- 📝 **Code Style**: Follow PEP 8 guidelines
- 📖 **Documentation**: Update README for new features
- 🧪 **Testing**: Include tests for new functionality
- 🔒 **Security**: Report vulnerabilities responsibly

### Areas for Contribution

| Area | Needs |
|------|-------|
| **Documentation** | Tutorials, API docs, translations |
| **Testing** | Unit tests, integration tests |
| **Features** | Accessibility improvements, UI enhancements |
| **Research** | Academic papers, use case studies |

---

## 📜 License

GhostMentor is released under the **GhostMentor Shadow License (GSL)**.

See [LICENSE.md](./LICENSE.md) for the complete license text.

### License Summary

- ✅ Free for educational and research use
- ✅ Open source with attribution requirements
- ✅ Modification allowed with license preservation
- ❌ Commercial use restrictions apply
- ❌ No warranty provided

---

## 🙏 Acknowledgments

### Technologies & Libraries

| Technology | Purpose | Link |
|------------|---------|------|
| **Google Gemini** | Multimodal AI processing | [AI Studio](https://aistudio.google.com/) |
| **Faster-Whisper** | Speech recognition | [GitHub](https://github.com/guillaumekln/faster-whisper) |
| **OpenCV** | Computer vision | [Website](https://opencv.org/) |
| **Pygame** | GUI and display | [Website](https://www.pygame.org/) |
| **PyAudio** | Audio capture | [Website](https://people.csail.mit.edu/hubert/pyaudio/) |

### Inspired By

This project was inspired by the open-source community's ongoing efforts to create accessible, AI-powered educational tools. We thank all contributors and researchers in the field of educational technology and assistive computing.

### Academic References

If you use GhostMentor in your research, please consider citing:

```bibtex
@software{ghostmentor2024,
  title = {GhostMentor: An AI-Powered Accessibility and Research Assistant},
  author = {maruf009sultan},
  year = {2024},
  url = {https://github.com/maruf009sultan/GhostMentor},
  note = {Educational and research tool for accessibility testing and AI assistance}
}
```

---

<div align="center">

## 📞 Support & Community

[![GitHub Issues](https://img.shields.io/badge/Support-GitHub%20Issues-blue?style=for-the-badge&logo=github)](https://github.com/maruf009sultan/GhostMentor/issues)
[![Discussions](https://img.shields.io/badge/Community-Discussions-purple?style=for-the-badge&logo=github)](https://github.com/maruf009sultan/GhostMentor/discussions)

---

### ⭐ Show Your Support

If GhostMentor has been helpful for your research or educational projects, please consider:

- ⭐ **Starring** this repository
- 🍴 **Forking** and contributing
- 📢 **Sharing** with the research community
- 📝 **Citing** in your publications

---

**Built with ❤️ for the Research & Education Community**

*"The code doesn't lie. Neither does GhostMentor."*

---

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Open Source Love](https://img.shields.io/badge/Open%20Source-💕-red?style=flat-square)](https://opensource.org/)

</div>
