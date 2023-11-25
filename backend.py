from datetime import datetime
from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import face_recognition

app = Flask(__name__)

data = []


def detect_faces(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)
    return len(face_landmarks_list)


@app.route('/analizar_imagen', methods=['POST'])
def analizar_imagen():
    # Recibir la imagen en base64 desde la solicitud POST
    data = request.get_json()
    id = data.get('id')
    timestamp = ('hora')
    imagen_base64 = data.get('imagen')

    # Decodificar la imagen base64 a formato de imagen
    imagen_decodificada = base64.b64decode(imagen_base64)
    np_arr = np.frombuffer(imagen_decodificada, np.uint8)
    imagen_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Detectar caras
    ahora = datetime.now()
    fecha_hora = ahora.strftime("%d/%m/%Y %H:%M")
    new_data = {'id': id,
                'numero de caras': detect_faces(imagen_cv),
                'hora enviada': timestamp,
                'hora analizada': fecha_hora}

    data.append(new_data)

    # Puedes enviar una respuesta con el resultado o simplemente confirmar la finalización del análisis
    return jsonify({'mensaje': 'Análisis completado'})


@app.route('/data', methods=['POST'])
def mostrar_data():
    msg = 'Data: ' + data
    return msg


if __name__ == '__main__':
    app.run(debug=True)