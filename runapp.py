
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import subprocess

import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
from PIL import Image
import tensorflow.compat.v1 as tf
from flask import Flask, render_template, Response, request, send_file
import db
import json
import uuid
import base64
from flask import request


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

modeldir = './model/20180402-114759.pb'
classifier_filename = './class/classifier.pkl'
npy = './npy'
train_img = "./train_img"
#cap = cv2.VideoCapture('http://61.7.241.156:8080/mjpg/video.mjpg')
cap = cv2.VideoCapture(0)


def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)
make_1080p()
font = cv2.FONT_HERSHEY_SIMPLEX
def process():
    people = []
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
            minsize = 60  # minimum size of face
            threshold = [0.7, 0.8, 0.8]  # three steps's threshold
            factor = 0.709  # scale factor
            margin = 44
            batch_size = 100  # 1000
            image_size = 182
            input_image_size = 160
            HumanNames = os.listdir(train_img)
            HumanNames.sort()
            print('Loading Model')
            facenet.load_model(modeldir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile, encoding='latin1')

            print('Start Camera')
            best_class_indices = None
            while True:
                ret, frame = cap.read()
                # frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)    #resize frame (optional)
                timer = time.time()
                if frame.ndim == 2:
                    frame = facenet.to_rgb(frame)
                bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                faceNum = bounding_boxes.shape[0]
                if faceNum > 0:
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(frame.shape)[0:2]
                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    for i in range(faceNum):
                        emb_array = np.zeros((1, embedding_size))
                        xmin = int(det[i][0])
                        ymin = int(det[i][1])
                        xmax = int(det[i][2])
                        ymax = int(det[i][3])
                        try:
                            if xmin <= 0 or ymin <= 0 or xmax >= len(frame[0]) or ymax >= len(frame):
                                print('Face is very close!')
                                continue
                            cropped.append(frame[ymin:ymax, xmin:xmax, :])
                            cropped[i] = facenet.flip(cropped[i], False)
                            scaled.append(np.array(Image.fromarray(cropped[i]).resize((image_size, image_size))))
                            scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                                   interpolation=cv2.INTER_CUBIC)
                            scaled[i] = facenet.prewhiten(scaled[i])
                            scaled_reshape.append(scaled[i].reshape(-1, input_image_size, input_image_size, 3))
                            feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)

                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            if best_class_probabilities > 0.10:
                                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # กรอบสีเขียวของใบหน้า
                                for H_i in HumanNames:
                                    if HumanNames[best_class_indices[0]] == H_i:
                                        result_names = HumanNames[best_class_indices[0]]
                                        # Xmin-1 แถบสีเขียว ด้านซ้าย ymin ล่างขึ้นบน #Xmax+1 แถบสีเขียวทางขวา
                                        cv2.rectangle(frame, (xmin - 1, ymin - 30), (xmax + 1, ymin - 2), (0, 255, 0), -1)
                                        cv2.putText(frame, result_names, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                                    1, (255, 255, 255), thickness=2)
                                        # Fontสีขาว ชื่อถูกต้อง
                                        # thickness=ความหน้า

                            else:
                                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                                cv2.rectangle(frame, (xmin, ymin - 20), (xmax, ymin - 2), (0, 0, 255), -1)
                                cv2.putText(frame, "Unknow", (xmin, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX,
                                            1, (255, 255, 255), thickness=2, lineType=2)
                        except:

                            print("error")

                endtimer = time.time()
                fps = 1 / (endtimer - timer)
                cv2.rectangle(frame, (15, 30), (135, 60), (255, 255, 255), -1)
                cv2.putText(frame, "fps: {:.2f}".format(fps), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                ret, buffer = cv2.imencode('.jpg', frame)  # compress and store image to memory buffer
                frame = buffer.tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                print(best_class_indices)
                print(HumanNames)
                if best_class_indices is not None:
                    name = HumanNames[best_class_indices[0]]
                for i, person in enumerate(people):
                    save_name, save_time = person

                    if name == save_name and faceNum > 0:
                        if time.time() - save_time > 5:
                            people[i] = (name, time.time())
                            matched = "{}|{:.2f}%".format(HumanNames[best_class_indices[0]], best_class_probabilities[0] * 100).split("|")
                            if best_class_probabilities < 0.10:
                                matched = "Unknow|Unknow".format(HumanNames[best_class_indices[0]], best_class_probabilities[0]).split("|")

                            filename = "%s.jpg" % str(uuid.uuid4())

                            b64 = base64.b64encode(frame)
                            decoded_data = base64.b64decode(b64)
                            np_data = np.frombuffer(decoded_data, np.uint8)
                            img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
                            cv2.imwrite("E:/ProjectFacenet/image/%s" % filename, img)

                            db.Insert(matched[0], matched[1], filename)
                        break

                else:
                    if faceNum > 0:
                        people.append((name, time.time()))
                        matched = "{}|{:.2f}%".format(HumanNames[best_class_indices[0]], best_class_probabilities[0] * 100).split("|")
                        if best_class_probabilities < 0.10:
                            matched = "Unknow|Unknow".format(HumanNames[best_class_indices[0]], best_class_probabilities[0]).split("|")
                        filename = "%s.jpg" % str(uuid.uuid4())

                        b64 = base64.b64encode(frame)
                        decoded_data = base64.b64decode(b64)
                        np_data = np.frombuffer(decoded_data, np.uint8)
                        img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
                        cv2.imwrite("E:/ProjectFacenet/image/%s" % filename, img)

                        db.Insert(matched[0], matched[1], filename)

                # print(people)
                key = cv2.waitKey(0)
                if key == 113:  # "q"
                    break
            cap.release()
            cv2.destroyAllWindows()

@app.route('/video_feed')
def video_feed():
    #Video streaming route
    return Response(process(), mimetype='multipart/x-mixed-replace; boundary=frame')

#@app.route('/query')
#def query():
    #start_date = request.args.get("start_date")
    #end_date = request.args.get("end_date")
    #result = list(map(lambda r: [r[0], r[1], r[2].strftime("%d-%b-%Y %H:%M:%S"), r[3], r[4]], db.Query(start_date, end_date)))
    #return Response(json.dumps(result), mimetype='application/json')
@app.route('/query')
def query():
    result = list(map(lambda r: [r[0], r[1], r[2].strftime("%d-%b-%Y %H:%M:%S"), r[3], r[4]], db.Query()))
    return Response(json.dumps(result), mimetype='application/json')

@app.route('/querydetails')
def querydetails():
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    result = list(map(lambda r: [r[0], r[1], r[2].strftime("%d-%b-%Y %H:%M:%S"), r[3], r[4]], db.Querydetails(start_date, end_date)))
    return Response(json.dumps(result), mimetype='application/json')


@app.route('/countrow')
def countrow():
    count = db.Countrows()
    return Response(json.dumps(count), mimetype='application/json')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/image', methods=['GET'])
def image():
    try:
        filename = request.args.get('filename')
        return send_file('image\\' + filename, mimetype='image/jpeg')
    except Exception as e:
        print(e)
        return Response('', mimetype='application/json')



if __name__ == "__main__":
    app.run(host='localhost', port='8080', debug=False, threaded=True)
