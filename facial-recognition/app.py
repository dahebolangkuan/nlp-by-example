import os
import logging
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, send, emit
from face_detector import FaceDetector

# logging.basicConfig(
#     level=logging.WARN,
#     format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
#     datefmt='%m-%d %H:%M')

app = Flask(__name__)
socketio = SocketIO(app)

face_detector = FaceDetector()

@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('video_feed')
def video_feed(data):
    res = face_detector.get_frame(data)
    socketio.emit('video_feed_response', res)


if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    socketio.run(app, host, port, debug=True)
