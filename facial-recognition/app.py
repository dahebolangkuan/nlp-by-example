import os
import logging
from flask import render_template, Response
from aiohttp import web
import socketio
from face_detector import FaceDetector

logging.basicConfig(
    level=logging.WARN,
    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M')

face_detector = FaceDetector()


async def index(req):
    index_file = open('templates/index.html')
    return web.Response(body=index_file.read().encode('utf-8'), headers={'content-type': 'text/html'})


app = web.Application()
app.add_routes([
    web.get('/', index)
])

sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')
sio.attach(app)


@sio.on('video_feed')
async def video_feed(req, data):
    res = face_detector.get_frame(data)
    await sio.emit('video_feed_response', res)

if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    web.run_app(app, host=host, port=port)
