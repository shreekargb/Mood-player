from flask import Flask, render_template_string, Response, jsonify,render_template
import cv2
from deepface import DeepFace
from collections import deque

app = Flask(__name__)
camera = cv2.VideoCapture(0)
emotion_buffer = deque(maxlen=10)
latest_emotion = "neutral" # Global variable to track emotion

def gen_frames():
    global latest_emotion
    while True:
        success, frame = camera.read()
        if not success: break
        
        try:
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            current_raw_emotion = results[0]['dominant_emotion']
            emotion_buffer.append(current_raw_emotion)
            from collections import Counter
            latest_emotion = Counter(emotion_buffer).most_common(1)[0][0]
            
            cv2.putText(frame, latest_emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except: pass

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # Simple HTML with a button
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_emotion')
def get_emotion():
    return jsonify({'emotion': latest_emotion})

if __name__ == '__main__':
    app.run(debug=True)