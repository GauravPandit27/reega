import cv2
import torch
from PIL import Image
import numpy as np
import wikipediaapi
import pyttsx3

def load_model():
    # Load YOLOv5 model
    return torch.hub.load('ultralytics/yolov5', 'yolov5s')

def create_wikipedia_api():
    # Create Wikipedia API object with a specified user agent
    return wikipediaapi.Wikipedia(
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent='my-cool-application/1.0'
    )

def capture_frame(cap):
    # Read frame from the camera
    ret, frame = cap.read()
    return ret, frame

def detect_objects(model, frame):
    # Convert the frame from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert the frame to PIL image
    pil_image = Image.fromarray(frame_rgb)
    # Perform inference
    return model(pil_image)

def display_info_on_frame(frame, detected_classes, frame_width):
    # Display detected classes in the top left corner
    info_text = ''
    for label in detected_classes:
        info_text += f'{label}\n'
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame

def display_frame(frame):
    # Display the frame
    cv2.imshow('Object Detection', frame)

def handle_key_press(key, paused, detected_classes, wiki_wiki):
    if key == ord('1'):
        paused = True
        # Print class name and information
        display_information(detected_classes, wiki_wiki)
    elif key == ord('2'):
        paused = True
        # Read only class name
        read_class_name(detected_classes)
    elif key == ord('4'):
        paused = True
        # Read whole information
        read_information(detected_classes, wiki_wiki)
    elif key == ord('3'):
        paused = False
    return paused

def display_information(detected_classes, wiki_wiki):
    print("Detected Classes:")
    for label in detected_classes:
        print(label)
        # Perform Wikipedia search for the detected object
        page = wiki_wiki.page(label)
        if page.exists():
            print("Summary:")
            summary = page.text.split('\n\n')[0]  # Get the first paragraph
            print(summary)
        else:
            print("No information found on Wikipedia")

def read_class_name(detected_classes):
    engine = pyttsx3.init()
    engine.say("the object Deceted is:")
    for label in detected_classes:
        engine.say(label)
    engine.runAndWait()

def read_information(detected_classes, wiki_wiki):
    engine = pyttsx3.init()
    for label in detected_classes:
        engine.say(label)
        # Perform Wikipedia search for the detected object
        page = wiki_wiki.page(label)
        if page.exists():
            summary = page.text  # Get the whole information
            engine.say(summary)
        else:
            engine.say("No information found on Wikipedia")
    engine.runAndWait()

def main():
    # Start capturing video from the default camera
    cap = cv2.VideoCapture(0)
    model = load_model()
    wiki_wiki = create_wikipedia_api()
    paused = False
    detected_labels = set()

    while True:
        if not paused:
            ret, frame = capture_frame(cap)
            if not ret:
                break
            
            results = detect_objects(model, frame)
            frame_height, frame_width = frame.shape[:2]
            detected_classes = set()
            
            for detection in results.pred[0]:
                label = model.names[int(detection[-1])]
                confidence = float(detection[4])

                if confidence > 0.7:  # Adjust confidence threshold as needed
                    detected_classes.add(label)

            frame = display_info_on_frame(frame, detected_classes, frame_width)
            display_frame(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
            paused = handle_key_press(key, paused, detected_classes, wiki_wiki)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
