from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import numpy as np
import time


app = Flask(__name__)
socketio = SocketIO(app)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

cap = cv2.VideoCapture(-1)
last_eye_detected_time = time.time()
cheating_count = 0

@app.route('/')
def index():
    return render_template('index.html')

def detect_cheating(frame):
    global last_eye_detected_time, cheating_count

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        last_eye_detected_time = time.time()
        cheating_popup_displayed = False

    for (x, y, w, h) in faces:
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) > 0:
            last_eye_detected_time = time.time()
            # cheating_popup_displayed = False

        # for (ex, ey, ew, eh) in eyes:
            # cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    

    if time.time() - last_eye_detected_time > 5 :
        socketio.emit('cheating_alert', {'message': f'This person is cheating! - {cheating_count}'})
        cheating_count += 1
        if cheating_count > 20:
            cap.release()
            cv2.destroyAllWindows()
    
    # cheating_popup_displayed = False

    if len(faces) > 1:
        socketio.emit('cheating_alert', {'message': f'Multiple faces detected! - {cheating_count}'})
        cheating_count += 1
        if cheating_count > 20:
            cap.release()
            cv2.destroyAllWindows()


    return frame

def generate_frames():
    
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = detect_cheating(frame)
        time.sleep(0.1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
    cap.release()
    cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

if __name__ == '__main__':
    app.run(debug=True)
