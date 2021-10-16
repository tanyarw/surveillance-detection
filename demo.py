from imageai import Detection
from twilio.rest import Client
import cv2
from dotenv import dotenv_values

config = dotenv_values(".env")  # from .env file, config = {"TWILIO_ACCOUNT_SID": "foo", "TWILIO_AUTH_TOKEN": "bar", "SENDER": "123456", "RECEIVER": "789123"}

account_sid = config['TWILIO_ACCOUNT_SID']
auth_token = config['TWILIO_AUTH_TOKEN']
sender = config['SENDER']
receiver = config['RECEIVER']
client = Client(account_sid, auth_token)

yolo = Detection.ObjectDetection()
yolo.setModelTypeAsYOLOv3()
yolo.setModelPath('./yolo.h5')
yolo.loadModel()


cam = cv2.VideoCapture(0) #0=front-cam
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 650)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 750)

custom = yolo.CustomObjects(bird=True,   cat=True,   dog=True,   horse=True,   sheep=True,   cow=True,   elephant=True,   bear=True,   zebra=True,  giraffe=True)

while True:

    ## read frames
    ret, img = cam.read()

    ## predict yolo
    img, preds = yolo.detectObjectsFromImage( 
        custom_objects=custom,
        input_image=img, 
        input_type="array",
        output_type="array",
        minimum_percentage_probability=70,
        display_percentage_probability=False,
        display_object_name=True)

    ## display predictions
    cv2.imshow("", img)

    if preds:
        print('Detected')
        print('.')
        print('.')
        print('.')
        print('.')
        print('.')
        print(preds[0]['name'])
        message = client.messages.create(
                     body="Hello! Our camera caught - " + preds[0]['name'] + "! Head over to the live stream to watch more. Regards, Team Wildsprint",
                     from_=sender,
                     to=receiver
                 )

        print(message.sid)    
        break


    ## press q or Esc to quit    
    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
        break

## close camera
cam.release()
cv2.destroyAllWindows()