from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

from classifier import training
from preprocess import preprocesses
from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def index():
  return render_template('control.html')

@app.route('/train')
def my_link():
  datadir = 'only_faceimages'
  modeldir = './model/20180402-114759.pb'
  classifier_filename = './class/classifier.pkl'
  print("Training Start")
  obj = training(datadir, modeldir, classifier_filename)
  get_file = obj.main_train()
  print('Saved classifier model to file "%s"' % get_file)
  sys.exit("All Done")


if __name__ == '__main__':

  app.run(host='localhost', port='8081', debug=False, threaded=True)