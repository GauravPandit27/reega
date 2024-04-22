# Object Detection with YOLOv5 and Wikipedia Integration

This Python script enables real-time object detection using YOLOv5 and provides additional information about the detected objects by fetching data from Wikipedia. The script utilizes computer vision libraries such as OpenCV and PyTorch, along with the Wikipedia API and text-to-speech capabilities.

## Installation

Before running the script, ensure you have the required dependencies installed. You can install them using pip:

```bash
pip install opencv-python torch torchvision pillow wikipedia-api pyttsx3
```

## Usage

1. Clone the repository or download the script from [GitHub]().

2. Run the script by executing the following command:

```bash
python object_detection.py
```

3. The script will start capturing video from the default camera and perform real-time object detection.

4. Detected objects will be displayed on the screen with their class names.

5. Press the following keys for additional functionalities:

   - Press `1` to display detailed information about the detected objects.
   - Press `2` to read only the class names of the detected objects.
   - Press `3` to resume real-time object detection if paused.
   - Press `4` to read the whole information about the detected objects.

6. To exit the script, press the `q` key.

## Acknowledgments

- YOLOv5: This script utilizes the YOLOv5 model for object detection. For more information about YOLOv5, refer to the [YOLOv5 GitHub repository](https://github.com/ultralytics/yolov5).

- Wikipedia API: The Wikipedia API is used to fetch information about detected objects from Wikipedia. For more information about the Wikipedia API, refer to the [Wikipedia-API documentation](https://pypi.org/project/Wikipedia-API/).

- Text-to-speech: The script uses the `pyttsx3` library for text-to-speech functionality. For more information about `pyttsx3`, refer to the [pyttsx3 documentation](https://pyttsx3.readthedocs.io/en/latest/).

## Notes

- Adjust confidence thresholds and other parameters as needed for optimal object detection performance.
- Ensure a stable internet connection for Wikipedia information retrieval.
- This script is intended for educational and experimental purposes and may require modifications for specific use cases or environments.
