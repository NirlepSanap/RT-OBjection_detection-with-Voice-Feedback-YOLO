# 🔍 Object Detection with Voice Feedback - YOLO v3 & gTTS

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v3-red.svg)](https://pjreddie.com/darknet/yolo/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build](https://img.shields.io/badge/Build-Passing-brightgreen.svg)](https://github.com/NirlepSanap/Object-Detection-with-Voice-Feedback-YOLO-v3-and-gTTS)

An intelligent **real-time object detection system** powered by **YOLO v3** that provides **spatial voice feedback** for detected objects. Perfect for accessibility applications, assistive technology, and computer vision enthusiasts.

## 🎯 Overview

This project combines computer vision with audio feedback to create an accessible object detection system. Using YOLO v3's state-of-the-art detection capabilities, it identifies objects in real-time and provides descriptive voice feedback about their spatial positions relative to the camera view.

## ✨ Key Features

- **🎯 Real-Time Detection**: Live object detection using webcam with YOLO v3
- **🗣️ Voice Feedback**: Spatial audio descriptions ("person on your left", "car in front of you")
- **📍 Position Mapping**: 9-zone spatial grid (top/mid/bottom × left/center/right)
- **📝 Text Recognition**: OCR capabilities using Tesseract for text detection
- **🖥️ Multiple Modes**: Static image, real-time video, and batch processing
- **📦 Executable**: Pre-built Windows executable for easy deployment
- **🎚️ Configurable**: Adjustable confidence thresholds and detection parameters
- **🔊 Smart Audio**: Cooldown system prevents audio spam, structured descriptions

## 🌟 Demo

### 🖼️ Detection Screenshot
<p align="center">
  <img src="example.jpeg" alt="Object Detection Demo" width="600">
  <br><em>Real-time object detection with bounding boxes and confidence scores</em>
</p>

### 🎬 Live Demo Video
**Voice Feedback Demo**: [Watch on Google Drive](https://drive.google.com/file/d/16qN5-FTHUVYb80j9rR6540ZUMIuCQYT9/view?usp=drive_link)

## 🛠️ Tech Stack

### Core Technologies
- **Computer Vision**: OpenCV 4.x, YOLO v3 (Darknet)
- **Deep Learning**: Pre-trained COCO dataset weights
- **Text-to-Speech**: gTTS (Google Text-to-Speech), pyttsx3
- **OCR**: Tesseract (pytesseract)
- **GUI**: OpenCV windows, real-time video display

### Development Tools
- **Language**: Python 3.8+
- **Packaging**: PyInstaller for executable creation
- **Version Control**: Git with Git LFS for large model files
- **Dependencies**: NumPy, threading, queue for async operations

## 📊 Architecture Overview

```
Input (Webcam/Image) → YOLO v3 Detection → Spatial Analysis → Voice Generation → Audio Output
                           ↓
                    Bounding Box Drawing → Display → User Interface
                           ↓
                    Text Recognition → OCR Processing → Console Output
```

### Detection Pipeline
1. **Input Processing**: Capture frame from webcam or load static image
2. **Preprocessing**: Resize and normalize image for YOLO input (416×416)
3. **Inference**: Forward pass through YOLO v3 network
4. **Post-processing**: Non-maximum suppression, confidence filtering
5. **Spatial Mapping**: Convert coordinates to spatial zones
6. **Voice Synthesis**: Generate descriptive audio feedback
7. **Display**: Render bounding boxes and labels on frame

## 📁 Project Structure

```
Object-Detection-with-Voice-Feedback-YOLO-v3-and-gTTS/
│
├── 📁 yolo/                          # YOLO model files
│   ├── yolov3.weights               # Pre-trained weights (248MB)
│   ├── yolov3.cfg                   # Network configuration
│   └── coco.names                   # Class labels (80 objects)
│
├── 📁 dist/                          # Executable builds
│   └── VisionSpeak.exe              # Windows executable (298MB)
│
├── 📄 main.py                       # Primary real-time detection script
├── 📄 ac.py                         # Advanced detection with OCR
├── 📄 script.py                     # Static image processing
├── 📄 main.spec                     # PyInstaller configuration
├── 📄 object_detection.spec         # Alternative build config
├── 📄 requirements.txt              # Python dependencies
├── 📄 tempCodeRunnerFile.py         # Development utilities
│
├── 📄 README.md                     # Project documentation
├── 🖼️ example.jpeg                  # Demo screenshot
├── 🎥 exa.mp4                       # Demo video
└── 🖼️ overview.png                  # Architecture diagram
```

## 🚀 Quick Start

### Prerequisites
```bash
# System Requirements
- Python 3.8 or higher
- Webcam (for real-time detection)
- Windows/Linux/macOS
- 4GB+ RAM (for YOLO model)
- Internet connection (for gTTS)
```

### Installation Methods

#### Method 1: Python Environment
```bash
# 1. Clone the repository
git clone https://github.com/NirlepSanap/Object-Detection-with-Voice-Feedback-YOLO-v3-and-gTTS.git
cd Object-Detection-with-Voice-Feedback-YOLO-v3-and-gTTS

# 2. Install dependencies
pip install opencv-python numpy pyttsx3 gtts playsound pytesseract

# 3. Download YOLO weights (if not included)
wget https://pjreddie.com/media/files/yolov3.weights -P yolo/

# 4. Install Tesseract (Windows)
# Download and install from: https://github.com/tesseract-ocr/tesseract
# Update path in ac.py: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

#### Method 2: Pre-built Executable (Windows)
```bash
# 1. Download the executable
# Navigate to dist/ folder and run VisionSpeak.exe
# No additional setup required!
```

### Quick Launch
```bash
# Real-time detection with voice feedback
python main.py

# Advanced detection with OCR
python ac.py

# Static image processing
python script.py -i path/to/image.jpg -y yolo/
```

## 💻 Usage Examples

### 1. Real-Time Object Detection
```python
# Basic real-time detection
python main.py
```
**Features:**
- Live webcam feed with bounding boxes
- Spatial voice feedback every 10 seconds
- Press 'q' to quit

### 2. Advanced Detection with OCR
```python
# Real-time with text recognition
python ac.py
```
**Output Examples:**
```
[DETECTED OBJECT] person
[DETECTED OBJECT] laptop
[TEXT DETECTED] Hello World
[VOICE FEEDBACK] person on your left. laptop in front of you.
```

### 3. Static Image Processing
```bash
# Process single image
python script.py -i example.jpeg -y yolo/ -c 0.5 -t 0.3
```

### 4. Using the Executable
```bash
# Windows users - just double-click
VisionSpeak.exe
```

## ⚙️ Configuration

### Detection Parameters
```python
# Confidence and NMS thresholds
confidence_threshold = 0.5  # Minimum detection confidence
nms_threshold = 0.3        # Non-maximum suppression threshold

# Audio settings
audio_cooldown = 10        # Seconds between voice announcements
speech_rate = 150         # Words per minute for pyttsx3
```

### Spatial Zone Configuration
```python
# 3x3 Grid mapping for voice feedback
zone_thresholds = [W // 3, (2 * W) // 3]  # Vertical divisions
height_zones = [H // 3, (2 * H) // 3]     # Horizontal divisions

# Voice descriptions
zone_text_map = {
    "left": "on your left",
    "center": "in front of you", 
    "right": "on your right"
}
```

### Customizing Object Classes
```python
# Edit yolo/coco.names to modify detectable objects
# Current classes: person, car, bicycle, etc. (80 total)
LABELS = open("yolo/coco.names").read().strip().split("\n")
```

## 🎯 Supported Object Classes

The system can detect **80 different object classes** from the COCO dataset:

<details>
<summary><strong>📋 Click to view all detectable objects</strong></summary>

**People & Animals:**
person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

**Vehicles:**
bicycle, car, motorbike, aeroplane, bus, train, truck, boat

**Indoor Objects:**
bottle, wine glass, cup, fork, knife, spoon, bowl, chair, sofa, bed, toilet, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors

**Outdoor Objects:**
traffic light, fire hydrant, stop sign, parking meter, bench, backpack, umbrella, handbag, tie, suitcase

**Sports & Recreation:**
frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket

**Food Items:**
banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake

*...and many more!*
</details>

## 🔊 Voice Feedback System

### Intelligent Audio Descriptions
```python
# Example voice outputs:
"person on your left"
"car in front of you"
"laptop and mouse on your right"
"person on your left. car in front of you. laptop on your right."
```

### Smart Audio Management
- **Cooldown System**: Prevents audio spam (10-second intervals)
- **Priority Queue**: Manages multiple audio requests
- **Structured Descriptions**: Organizes objects by spatial zones
- **Duplicate Prevention**: Avoids repetitive announcements

## 📊 Performance Metrics

### System Requirements
- **Minimum RAM**: 4GB (8GB recommended)
- **Processing Speed**: ~15-30 FPS (depending on hardware)
- **Model Size**: 248MB (YOLO weights)
- **Executable Size**: 298MB (includes all dependencies)

### Detection Accuracy
- **mAP (COCO)**: 57.9% (YOLO v3 baseline)
- **Confidence Threshold**: 0.5 (adjustable)
- **Supported Resolution**: Up to 1080p input
- **Processing Time**: ~30-100ms per frame

## 🔧 Advanced Features

### 1. Multi-Threading Architecture
```python
# Separate threads for audio and detection
threading.Thread(target=speak_worker, daemon=True).start()
audio_queue = queue.Queue()  # Async audio processing
```

### 2. Dynamic Resource Management
```python
# PyInstaller resource handling
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # PyInstaller temp folder
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
```

### 3. OCR Integration
```python
# Tesseract text recognition
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
extracted_text = pytesseract.image_to_string(gray)
```

## 🚀 Deployment Options

### 1. Development Environment
```bash
# Clone and run directly
git clone <repository>
pip install -r requirements.txt
python main.py
```

### 2. Executable Distribution
```bash
# Build executable using PyInstaller
pyinstaller main.spec
# Output: dist/VisionSpeak.exe
```

### 3. Docker Container
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN apt-get update && apt-get install -y \
    libopencv-dev \
    tesseract-ocr \
    && pip install -r requirements.txt

CMD ["python", "main.py"]
```

## 🧪 Testing & Validation

### Test Scenarios
```bash
# 1. Single object detection
python script.py -i test_images/single_person.jpg -y yolo/

# 2. Multiple objects
python script.py -i test_images/street_scene.jpg -y yolo/

# 3. Low-light conditions
python main.py  # Test with dim lighting

# 4. OCR functionality  
python ac.py    # Test with text-containing images
```

### Performance Benchmarks
- **Accuracy**: Test against COCO validation set
- **Speed**: Measure FPS across different hardware
- **Audio Latency**: Time from detection to voice output
- **Memory Usage**: Monitor RAM consumption during execution

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Development Setup
```bash
# 1. Fork and clone
git clone https://github.com/yourusername/Object-Detection-with-Voice-Feedback.git

# 2. Create feature branch
git checkout -b feature/amazing-feature

# 3. Make changes and test
python main.py  # Test your changes

# 4. Commit and push
git commit -m "Add amazing feature"
git push origin feature/amazing-feature
```

### Contribution Areas
- **🎯 Model Improvements**: YOLO v4/v5 integration, custom training
- **🔊 Audio Enhancements**: Multiple TTS engines, language support
- **📱 Platform Support**: Mobile apps, web interface
- **🎨 UI/UX**: GUI improvements, settings panel
- **🔧 Performance**: Optimization, GPU acceleration
- **📚 Documentation**: Tutorials, API docs, examples

## 🔮 Roadmap & Future Enhancements

### Version 2.0 Goals
- [ ] **YOLO v5/v8 Integration**: Improved accuracy and speed
- [ ] **Multi-language Support**: TTS in multiple languages
- [ ] **Custom Model Training**: Train on specialized datasets
- [ ] **Mobile App**: iOS/Android applications
- [ ] **Cloud API**: RESTful API for remote processing
- [ ] **GUI Interface**: Desktop application with settings
- [ ] **3D Spatial Audio**: Directional audio feedback
- [ ] **Integration APIs**: Webhooks, third-party integrations

### Long-term Vision
- **Accessibility Platform**: Complete assistive technology suite
- **Edge Computing**: Optimized for embedded devices
- **AR/VR Integration**: Mixed reality applications
- **IoT Connectivity**: Smart home integration

## 🐛 Troubleshooting

### Common Issues

#### 1. **"No module named 'cv2'"**
```bash
pip install opencv-python
```

#### 2. **YOLO weights not found**
```bash
# Download weights manually
wget https://pjreddie.com/media/files/yolov3.weights -P yolo/
```

#### 3. **Tesseract not found (Windows)**
```python
# Install Tesseract and update path in ac.py
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

#### 4. **Audio not working**
```bash
# Install audio dependencies
pip install pyttsx3 gtts playsound
```

#### 5. **Low FPS/Performance**
- Reduce input resolution
- Increase confidence threshold
- Close other applications
- Use GPU acceleration (if available)

### Debug Mode
```python
# Enable verbose logging
print(f"[DEBUG] Detection time: {end - start:.6f} seconds")
print(f"[DEBUG] Objects detected: {len(idxs)} objects")
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License - Feel free to use, modify, and distribute
Attribution required for commercial use
```

## 👨‍💻 Author & Maintainer

**Nirlep Sanap**
- 🐱 GitHub: [@NirlepSanap](https://github.com/NirlepSanap)
- 💼 LinkedIn: [Connect with me](https://linkedin.com/in/nirlepsanap)
- 📧 Email: contact@nirlepsanap.com

## 🙏 Acknowledgments & Credits

### Technologies & Frameworks
- **[YOLO v3](https://pjreddie.com/darknet/yolo/)**: Joseph Redmon & Ali Farhadi
- **[OpenCV](https://opencv.org/)**: Computer vision library
- **[COCO Dataset](https://cocodataset.org/)**: Common Objects in Context
- **[Google Text-to-Speech](https://cloud.google.com/text-to-speech)**: gTTS API
- **[Tesseract OCR](https://github.com/tesseract-ocr/tesseract)**: Google's OCR engine

### Inspiration & References
- Accessibility technology research
- Computer vision for assistive devices
- Real-time object detection applications
- Voice user interface design

## 📞 Support & Community

### Getting Help
1. **📖 Documentation**: Check this README and code comments
2. **🐛 Issues**: [Create GitHub issue](https://github.com/NirlepSanap/Object-Detection-with-Voice-Feedback/issues)
3. **💬 Discussions**: [GitHub Discussions](https://github.com/NirlepSanap/Object-Detection-with-Voice-Feedback/discussions)
4. **📧 Contact**: Reach out via LinkedIn for urgent matters

### Community Guidelines
- Be respectful and inclusive
- Provide detailed bug reports with logs
- Share your use cases and improvements
- Help others learn and contribute

---

<div align="center">

**⭐ Star this repository if it helped you! ⭐**

<br>

**🔗 Share with the community • 🤝 Contribute code • 📝 Report issues**

<br>

*Made with ❤️ for accessibility and computer vision*

<br>

![Visitors](https://visitor-badge.glitch.me/badge?page_id=nirlepsanap.object-detection-voice-feedback)

</div>