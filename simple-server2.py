from flask import Flask, render_template, Response, send_file, request, jsonify
from picamera2 import Picamera2
import cv2
import threading

from typing import Tuple
from cv2.typing import MatLike, Point
import numpy as np

import serial

from time import sleep

app = Flask(__name__)

# Initialize the camera
camera = Picamera2()
camera.configure(camera.create_preview_configuration(
    main={"size": (1920, 1080)},
#    transform=Picamera2.Transformation(270)  # Rotate the image by 270 degrees DOESNT WORK
))
camera.start()


video_filename = 'output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify codec
out = cv2.VideoWriter(video_filename, fourcc, 30.0, (1080, 1920))  # Output file, FPS, and frame size

current_frame = None
current_detection_frame = None
current_threshold_mask = None

center_object = (480 / 2, 640 / 2)

# Shared variables for storing filter settings
filter_settings = {
    "hue": (72, 106),
    "saturation": (0, 255),
    "lightness": (0, 255),
    "blur": 10
}


def load_frame():
    global current_frame
    print('starting camera')
    while True:
        # Capture an image as a numpy array
        frame = camera.capture_array()
        
        # Convert image to BGR format used by OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.flip(frame, 0)
        
        # Rotate the image 270 degrees
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Write the frame to the video file
        # out.write(frame)
        current_frame = frame


def analyze_frame():
    global current_detection_frame, current_threshold_mask
    while True:
        sleep(0.05)
        if current_frame is None:
            print('cam not started yet')
            continue
            
        downsized_frame = cv2.resize(current_frame, (480, 640), interpolation=cv2.INTER_AREA)
        
        preview, mask = generate_mask_frame(downsized_frame)
        # current_detection_frame = cv2.resize(frame, (1080, 1920), interpolation=cv2.INTER_AREA)
        current_detection_frame = preview
        current_threshold_mask = mask
        

def generate_frames():
    while True:
        sleep(0.05)
        if current_frame is None or current_threshold_mask is None or current_detection_frame is None:
            print('cam not started yet')
            continue
        
        # Encode the frame in JPEG format
        frame_resized = cv2.resize(current_frame, (480, 640))
        
        mask_colored = cv2.cvtColor(current_threshold_mask, cv2.COLOR_GRAY2BGR)
                
        combined_frame = np.hstack((frame_resized, mask_colored, current_detection_frame))
        
        ret, buffer = cv2.imencode('.jpg', combined_frame)
        frame = buffer.tobytes()
        
        # Yield the image bytes as part of a multipart HTTP response
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    
def move_stepper():
    # Open serial connection
    mask_width = 480
    mask_height = 640
    middle_x, middle_y = (mask_width / 2, mask_height / 2)
    
    boundary_x = 10
    
    try:
        with serial.Serial('/dev/ttyACM0', baudrate=9600, timeout=1) as ser:
            while True:
                object_x, object_y = center_object
                
                object_x -= middle_x
                object_y -= middle_y
                
                object_x *= 100 / middle_x
                
                move_command = f'{int(object_x / 20)},0,0,SYN\n'
                
                if object_x > boundary_x or object_x < -boundary_x:
                    ser.write(move_command.encode('ascii'))
                else:
                    continue
                ser.flush()
                    
                # Example: Read a response
                response = ser.readline()
                if response != b'ACK\r\n':
                    print(f'Got faulty response: {response}')
                    
                sleep(0.01)
        
    except serial.SerialException as e:
        print(f"Serial Exception: {e}")
    
    except Exception as e:
        # Catch any other exceptions
        print(f"An unexpected error occurred: {e}")
    print('Closed ACM0!')


def get_pos(mask: MatLike) -> Point:
    center = [np.average(indices) for indices in np.where(mask >= 1)]
    center.reverse()
    if np.isnan(center).any():
        return [-10, -10]
    return list(map(round, center))


def get_connected_components(mask: MatLike) -> Tuple:
    output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    return output

def draw_boundaries_biggest_component(components, image):
    global center_object
    (numLabels, labels, stats, centroids) = components
    # loop over the number of unique connected component labels
    biggest = 0
    biggest_val = 0
    for i in range(0, numLabels):
        # if this is the first component then we examine the
        # *background* (typically we would just ignore this
        # component in our loop)
        if i == 0:
            # print(f"examining component {i + 1}/{numLabels} (background)")
            continue
        # otherwise, we are examining an actual connected component
        # else:
        #     print(f"examining component {i + 1}/{numLabels}")
        # extract the connected component statistics and centroid for
        # the current label
        area = stats[i, cv2.CC_STAT_AREA]
        if area > biggest_val:
            biggest = i
            biggest_val = area

    i = biggest
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    (cX, cY) = centroids[i]
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.circle(image, (int(cX), int(cY)), 4, (0, 0, 255), -1)
    center_object = centroids[i]


def generate_mask_frame(frame):
    blur_radius = filter_settings['blur']
    low_H, high_H = filter_settings['hue']
    low_S, high_S = filter_settings['saturation']
    low_L, high_L = filter_settings['lightness']
    
    frame = frame.copy()
    original = frame.copy()
    frame = cv2.GaussianBlur(frame, (blur_radius*2+1, blur_radius*2+1), 10)

    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(frame_HSV, (low_H, low_S, low_L), (high_H, high_S, high_L))

    object = get_pos(frame_threshold)
    cv2.circle(frame, object, 10, (0, 255, 0), 4)
    cv2.circle(frame_threshold, object, 10, (0, 255, 0), 4)

    components = get_connected_components(frame_threshold)
    draw_boundaries_biggest_component(components, original)
    draw_boundaries_biggest_component(components, frame_threshold)

    return original, frame_threshold


@app.route('/')
def index():
    return render_template('index.html', ctx=filter_settings)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video.mp4')
def saved_video():
    try:
        return send_file(video_filename, as_attachment=True)
    except Exception as e:
        return str(e)

@app.route('/update_filters', methods=['POST'])
def update_filters():
    data = request.get_json()
    for key in filter_settings.keys():
        if key in data:
            filter_settings[key] = data[key]
    return jsonify({'status': 'success', 'filter_settings': filter_settings})

if __name__ == '__main__':
    try:
        thread = threading.Thread(target=load_frame)
        thread.start()
        
        thread_analyze_video = threading.Thread(target=analyze_frame)
        thread_analyze_video.start()
        
        thread_stepper_mover = threading.Thread(target=move_stepper)
        thread_stepper_mover.start()
        app.run(host='0.0.0.0', port=5000)
    finally:
        camera.stop()
        out.release()
