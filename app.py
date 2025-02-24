from flask import Flask, render_template, request, Response
import cv2
import dlib
import time
import requests
from scipy.spatial import distance
from twilio.rest import Client
import os
from dotenv import load_dotenv

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Twilio credentials (Set in Railway ENV variables)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE = os.getenv("TWILIO_PHONE")

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Load dlib model (Change path if needed)
FILE_ID = "1_WaKEZwuFDbJ7Hjh4J1wmiuR1CpVHih-"
DEST_PATH = "shape_predictor_68_face_landmarks.dat"

if not os.path.exists(DEST_PATH):
    print("Downloading shape predictor model...")
    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    response = requests.get(url, stream=True)
    with open(DEST_PATH, "wb") as file:
        for chunk in response.iter_content(1024):
            file.write(chunk)
    print("✅ Model downloaded successfully.")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(DEST_PATH)

contact_info = {}


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def send_alert():
    """Send an SMS alert via Twilio"""
    if "phone" in contact_info:
        try:
            client.messages.create(
                body="🚨 Drowsiness detected! Please check on the driver.",
                from_=TWILIO_PHONE,
                to=contact_info["phone"],
            )
            print("✅ SMS alert sent!")
        except Exception as e:
            print(f"❌ Error sending SMS: {e}")


def generate_frames():
    """Live video feed & drowsiness detection."""
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            left_eye = [
                (landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)
            ]
            right_eye = [
                (landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)
            ]
            avg_EAR = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

            if avg_EAR < 0.25:
                send_alert()
                cv2.putText(
                    frame,
                    "DROWSY!",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 0, 255),
                    3,
                )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    camera.release()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
