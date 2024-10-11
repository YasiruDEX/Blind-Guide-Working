import requests
import pyttsx3
from django.http import JsonResponse
from threading import Thread, Event
import time
import cv2 as cv
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from openai import OpenAI

# Initialize TTS engine
tts_engine = pyttsx3.init()

# Point to the local server
client = OpenAI(base_url="http://192.168.137.1:1234/v1", api_key="lm-studio")

# Function to call the LLM and generate a story from the image captions
def generate_story_from_captions():
    # Read the captions from the output.txt file
    with open("output.txt", "r") as file:
        captions = file.read()

    # Prepare the system prompt with the captions
    system_prompt = f"""
    You are an intelligent assistant that generates short, coherent stories from image captions. 
    The captions describe scenes from video frames that were captured in sequence. Based on the captions provided, 
    create a story that is no more than 50 words. Ensure the story connects the events in a logical and descriptive manner.

    Here are the captions:
    {captions}

    Now, generate a story of no more than 50 words based on the captions provided.
    """

    # Call the LLM to generate the story
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
        ],
        temperature=0.7,
        max_tokens=50
    )

    # Return the generated story
    return completion.choices[0].message.content

# Function to use the remote server for audio playback
def narrate_story(story):
    # Define the remote server URL that handles audio playback
    remote_audio_url = "http://[2402:4000:11c0:def5:6414:e448:de1:25fa]:5000/transcribe"

    # Send the story to the server, assuming the server will convert it to audio and play it
    response = requests.post(remote_audio_url, json={"text": story})

    # Check the response from the server
    if response.status_code == 200:
        print("Audio played successfully on the remote server.")
    else:
        print(f"Error: {response.status_code}, {response.text}")

# Initialize the BLIP processor and model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Global variables to manage status and background thread
status = False
thread = None
stop_event = Event()

# Function to run the image captioning script
def image_transcription_task():
    print("Starting image transcription task...")  # Debugging
    
    # URL for the ESP32 stream (replace with your stream URL)
    url = "http://192.168.137.139:81/stream"

    # Create a VideoCapture object for the stream
    capture = cv.VideoCapture(url)
    
    print("Starting...")  # Debugging

    # Check if the stream is opened correctly
    if not capture.isOpened():
        print("Error: Unable to open video stream")
        return
    
    frame_count = 0
    frame_skip = 5  # Number of frames to skip before generating a caption
    
    # Clear the file before starting
    with open("output.txt", "w") as file:
        pass  # This just opens the file in write mode, which clears it

    while not stop_event.is_set():  # Check if the stop_event flag is set
        ret, frame = capture.read()

        if ret:
            print("Frame captured successfully.")  # Debugging
            
            # Resize the frame to make processing faster
            small_frame = cv.resize(frame, (320, 240))

            # Only process every nth frame to avoid buffering
            if frame_count % frame_skip == 0:
                # Convert the OpenCV frame (BGR) to PIL image (RGB)
                pil_image = Image.fromarray(cv.cvtColor(small_frame, cv.COLOR_BGR2RGB))

                # Unconditional image captioning
                inputs = processor(pil_image, return_tensors="pt")
                out = model.generate(**inputs)

                # Decode the generated caption
                caption = processor.decode(out[0], skip_special_tokens=True)
                print(f"Generated caption: {caption}")  # Debugging
                
                # Write the caption to output.txt
                with open("output.txt", "a") as file:
                    file.write(f"frame {frame_count} : {caption}\n")

            frame_count += 1
        
        else:
            print("Failed to capture frame.")  # Debugging
        
        time.sleep(0.2)  # Pause for 0.2 seconds between frames
    
    # Release the video capture object when stopped
    print("Releasing video capture.")  # Debugging
    capture.release()

# Start the transcription task
def start(request):
    global status, thread, stop_event
    if not status:
        status = True
        stop_event.clear()
        print("Starting transcription task...")  # Debugging
        # Start the background thread
        thread = Thread(target=image_transcription_task)
        thread.start()
    return JsonResponse({'status': status, 'message': 'Started transcription task'})

# Stop the transcription task and narrate the generated story
def stop(request):
    global status, stop_event
    if status:
        status = False
        stop_event.set()  # Signal the background thread to stop
        if thread:
            thread.join()  # Wait for the thread to finish
        print("Stopped transcription task.")  # Debugging

        # Call the function to generate a story from the captions
        story = generate_story_from_captions()

        # Narrate the generated story using the remote server
        narrate_story(story)

        return JsonResponse({'status': status, 'message': 'Stopped transcription task', 'story': story})
    
    return JsonResponse({'status': status, 'message': 'Transcription task was not running'})
