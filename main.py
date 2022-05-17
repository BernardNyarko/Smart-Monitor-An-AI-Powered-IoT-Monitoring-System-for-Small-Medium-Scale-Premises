# Libraries for Flask web frame, face recognition, tensorflow custom detection and OpenCV
from flask import Flask, render_template, Response, request
import time
from time import sleep
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #suppresses tensorflow logs
import face_recognition
import numpy as np
import cv2

import six.moves.urllib as urllib
import sys
import tarfile
import zipfile
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
import tensorflow as tf

from utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

# Libraries for sending email
import smtplib
import imghdr
from email.message import EmailMessage

#Libraries  for buzzer, sensor, LED, and button
from gpiozero import Buzzer
from gpiozero import Button
from gpiozero import MotionSensor
from gpiozero import LED

# Library for using multiple Threads
import threading

#Libraries for retrieving the path of Images folder to count the number of images
import os, os.path



#variable declarations
video_capture = cv2.VideoCapture(-1)
width= int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
save_recording = cv2.VideoWriter('/home/pi/pi-camera-stream-flask/videos/' + (time.strftime("%y%b%d_%H:%M:%S")) + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (width,height))
images_directory_path = '/home/pi/pi-camera-stream-flask/images'
list_of_names = []
list_of_images = []
photo_detections = []
alert_buffer = []
timer = 300
a = 0
e = 0
w = 0
buzzer = Buzzer(17)
button = Button(21)
pir = MotionSensor(4)
red = LED(25)
yellow = LED(8)
green = LED(7)

#Function to send alerts (emails/text messages)
def alerts(subject, body, to):
    msg = EmailMessage()
    msg.set_content(body)
    msg['subject'] = subject
    msg['to'] = to
    my_email_address = "kofinyarko42@gmail.com"
    msg['from'] = my_email_address
    password = "tfeqvtkkeulctrfh"
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(my_email_address, password)
    server.send_message(msg)
    server.quit()

"""Function to send alerts with image attachments after running
   custom tensorflow model for detections on images taken when PIR sensor is triggered"""
def alerts_with_attachment(subject, body, to):
    message = EmailMessage()
    message.set_content(body)
    message['subject'] = subject
    message['to'] = to
    my_email_address = "kofinyarko42@gmail.com"
    message['from'] = my_email_address
    password = "tfeqvtkkeulctrfh"
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    attached_images = ['/home/pi/pi-camera-stream-flask/image_detections/' + photo_detections[0], '/home/pi/pi-camera-stream-flask/image_detections/' + photo_detections[1], '/home/pi/pi-camera-stream-flask/image_detections/' + photo_detections[2]]
    for picture in attached_images:
        with open(picture, 'rb') as file:
            image_data = file.read()
            file_type = imghdr.what(file.name)
            name_of_file = file.name
        message.add_attachment(image_data, maintype = 'image', subtype = file_type, filename = name_of_file)
    server.login(my_email_address, password)
    server.send_message(message)
    server.quit()

#function to run tensorflow custom detection model on images
def custom_detection_model():
    # patch tf1 into `utils.ops`
    utils_ops.tf = tf.compat.v1
    # Patch the location of gfile
    tf.gfile = tf.io.gfile

    #Variable declarations
    global list_of_images
    global a
    images = 0
    number = 0
    directory_path = '/home/pi/pi-camera-stream-flask/image_detections'
    for detected in os.listdir(directory_path):
        path = os.path.join(directory_path, detected)
        if os.path.isfile(path):
            images+=1
    if images !=0:
        number=images


    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = '/home/pi/models/research/object_detection/custom_data_detection/saved_model/label_map.pbtxt'
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    print('loading custom tensorflow model, please wait...')
    detection_model = tf.saved_model.load('/home/pi/models/research/object_detection/custom_data_detection/saved_model')
    print('custom tensorflow model loaded successfully')

    def run_inference_for_single_image(model, image):
      image = np.asarray(image)
      # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
      input_tensor = tf.convert_to_tensor(image)
      # The model expects a batch of images, so add an axis with `tf.newaxis`.
      input_tensor = input_tensor[tf.newaxis,...]

      # Run inference
      model_fn = model.signatures['serving_default']
      output_dict = model_fn(input_tensor)
      num_detections = int(output_dict.pop('num_detections'))
      output_dict = {key:value[0, :num_detections].numpy() 
                     for key,value in output_dict.items()}
      output_dict['num_detections'] = num_detections

      # detection_classes should be ints.
      output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
      if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                  output_dict['detection_masks'], output_dict['detection_boxes'],
                   image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.8,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
        
      return output_dict


    def show_inference(model, image_np):
      output_dict = run_inference_for_single_image(model, image_np)
      image_with_detections =vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              output_dict['detection_boxes'],
              output_dict['detection_classes'],
              output_dict['detection_scores'],
              category_index,
              instance_masks=output_dict.get('detection_masks_reframed', None),
              use_normalized_coordinates=True,
              line_thickness=4)
      return image_with_detections


    while True:
        if a ==1:
            sleep(3)
            for z in range(3):
                photo_path = '/home/pi/pi-camera-stream-flask/images/'
                img = os.path.join(photo_path,list_of_images[z])
                img = cv2.imread(img)
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                final_img = show_inference(detection_model,img)
                final_img = cv2.cvtColor(final_img,cv2.COLOR_RGB2BGR)
                #cv2.imshow('image',final_img)
                numbers = str(number)
                cv2.imwrite('/home/pi/pi-camera-stream-flask/image_detections/photo_'+ numbers + '.jpg',final_img)
                photo_detections.append('photo_'+ numbers + '.jpg')
                number+=1
            if list_of_names: # if list is not empty
                    list_to_string = ', '.join([str(element) for element in list_of_names])# converting list to string
                    alerts_with_attachment("SmartMonitor Alert", "Motion sensor triggered !!!\n Likely causes include " + " " + list_to_string + ". " + " Please check cameras", "8322491151@tmomail.net")
                    list_of_images.clear()
                    photo_detections.clear()
                    z = 0
            else:# if model cannot tell possible cause of triggering sensor
                alerts_with_attachment("SmartMonitor Alert", "Caution!! Caution!! Caution!!\n Motion sensor triggered !!! No likely causes found \nPlease check cameras", "8322491151@tmomail.net")    
                list_of_images.clear()
                photo_detections.clear()
                z = 0

    
#Function to set timer
def counter():
    global timer
    global list_of_names
    for x in range(timer):
        timer = timer - 1
        sleep(1)
    list_of_names.clear()
    alert_buffer.clear()
    

# Function for motion sensor
def sensor_triggered():
    global a
    while True:
        pir.wait_for_motion()
        a=1
        print("Motion Detected!!!")
        sleep(3)
        a=0
        pir.wait_for_no_motion
        for y in range(10):
            buzzer.on()
            sleep(1)
            buzzer.off()
            sleep(1)

#function to turn on LED
def LED():
    global a
    global e
    while True:
        if a==1:
            green.off()
            yellow.on()
            sleep(2.8)
            yellow.off()
            for d in range(10):
                red.on()
                sleep(1)
                red.off()
                sleep(1)
            yellow.on()
            sleep(2)
            yellow.off()
        if e==1:
            green.off()
            yellow.on()
            sleep(1)
            yellow.off()
            for d in range(10):
                red.on()
                sleep(1)
                red.off()
                sleep(1)
            yellow.on()
            sleep(2)
            yellow.off()
        else:
            green.on()
            sleep(1)
            green.off()
            sleep(1)
        
# Button to manually trigger buzzer
def button_pressed():
    global e
    while True:
        button.wait_for_press()
        e=1
        sleep(2)
        for z in range(10):
            buzzer.on()
            sleep(1)
            buzzer.off()
            sleep(1)
        e=0


# Thread for running tensorflow model on images
tensorflow_model_thread = threading.Thread(target = custom_detection_model)
tensorflow_model_thread.start()

# Thread for checking motion with PIR sensor
sensor_thread = threading.Thread(target = sensor_triggered)
sensor_thread.start()

# Thread for button when pressed
button_thread = threading.Thread(target = button_pressed)
button_thread.start()

# Thread for starting timer
timer_thread = threading.Thread(target = counter)
timer_thread.start()

# Thread for changing LED (red, yellow and green
LED_thread = threading.Thread(target = LED)
LED_thread.start()


# App Globals (do not edit)
app = Flask(__name__)

# Load a sample picture and learn how to recognize it.
ahmed_image = face_recognition.load_image_file("/home/pi/pi-camera-stream-flask/Ahmed/ahmed.jpg")
ahmed_face_encoding = face_recognition.face_encodings(ahmed_image)[0]

# Load a second sample picture and learn how to recognize it.
bellam_image = face_recognition.load_image_file("/home/pi/pi-camera-stream-flask/bellam/bellam.jpg")
bellam_face_encoding = face_recognition.face_encodings(bellam_image)[0]

# Load a second sample picture and learn how to recognize it.
bernard_image = face_recognition.load_image_file("/home/pi/pi-camera-stream-flask/Bernard/bernard.jpg")
bernard_face_encoding = face_recognition.face_encodings(bernard_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    ahmed_face_encoding,
    bellam_face_encoding,
    bernard_face_encoding
]
known_face_names = [
    "Ahmed",
    "Bellam",
    "Bernard"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

@app.route('/')
def index():
    return render_template('index.html') #you can customze index.html here

def gen():
    process_this_frame = True
    f=0
    b=0
    g=0
    number_of_images = 0
    for image in os.listdir(images_directory_path):
        image_path = os.path.join(images_directory_path, image)
        if os.path.isfile(image_path):
            number_of_images+=1
    if number_of_images !=0:
        g=number_of_images
    
    #get camera frame
    while True:
        ret, frame = video_capture.read()
        
        #saving frames in video format before performing face recognition
        save_recording.write(frame)
        
        #capturing images if PIR sensor is triggered
        if a==1:
            for b in range(3):
                if b==f:
                    cv2.imwrite('/home/pi/pi-camera-stream-flask/images/image_' + str(g) +'.jpg',frame)
                    list_of_images.append('image_' + str(g) + '.jpg')
                    f+=1
                    g+=1
            
        else:
            f=0
            b=0

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    if name not in list_of_names:
                        list_of_names.append(name)
                
                # Conditional statements to send alerts   
                if name in list_of_names and name not in alert_buffer and name != "Unknown":# alert for known persons
                    alerts("SmartMonitor Alert", name + " " + "is at the Premises", "8322491151@tmomail.net")
                    alert_buffer.append(name)
                if name in list_of_names and name not in alert_buffer and name == "Unknown": # alert for unknown persons
                    alerts("SmartMonitor Alert", name + " " + "person is at the Premises", "8322491151@tmomail.net")
                    alert_buffer.append(name)

                face_names.append(name)

        process_this_frame = not process_this_frame


        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 1)

            # Draw a label with a name below the face
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, name, (left + 1, top - 7), font, 0.5, (0, 255, 255), 1)

        ret, buffer = cv2.imencode('.jpg',frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':

    app.run(host='0.0.0.0', debug=False)