from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import face_recognition
import pickle
import cv2
import os
import glob
from sklearn.metrics import precision_score, accuracy_score
from sklearn.metrics import confusion_matrix

import cv2
import numpy as np
import facenet
import detect_face
import os
import pickle
from PIL import Image
import tensorflow.compat.v1 as tf
import time

start = time.process_time()
modeldir = './model/20180402-114759.pb'
classifier_filename = './class/classifier.pkl'
npy = './npy'
train_img = "./train_img"
font = cv2.FONT_HERSHEY_SIMPLEX
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
def findPerson(frame):
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
			best_class_probabilities = predictions[
				np.arange(len(best_class_indices)), best_class_indices]
			if best_class_probabilities > 0.10:
				cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # กรอบสีเขียวของใบหน้า
				for H_i in HumanNames:
					if HumanNames[best_class_indices[0]] == H_i:
						result_names = HumanNames[best_class_indices[0]]
						return HumanNames[best_class_indices[0]]

def runDetection():
	names = []
	test = []
	for filepath in glob.iglob(r'E:\ProjectFacenet\Test\*\*.jpg'):
		x = str(filepath).split("\\")[-2]
		test.append(x)
		print(filepath)
		a = cv2.imread(filepath)

		name = findPerson(a)
		names.append(name)


	confusion_matrix(test, names)
	end = time.process_time()
	print(accuracy_score(test, names))
	print(f"Runtime of the program is {end - start}")

if __name__ == "__main__":
	runDetection()