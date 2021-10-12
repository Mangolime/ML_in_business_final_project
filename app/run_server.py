# USAGE
# Start the server:
# 	python run_front_server.py
# Submit a request via Python:
#	python simple_request.py

# import the necessary packages
import dill
import pandas as pd
import os
dill._dill._reverse_typemap['ClassType'] = type
#import cloudpickle
import flask
import logging
from logging.handlers import RotatingFileHandler
from time import strftime

# initialize our Flask application and the model
app = flask.Flask(__name__)
model = None

handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def load_model(model_path):
	# load the pre-trained model
	global model
	with open(model_path, 'rb') as f:
		model = dill.load(f)
	print(model)

modelpath = "/app/app/models/rf_pipeline.dill"
load_model(modelpath)

@app.route("/", methods=["GET"])
def general():
	return """Welcome to fraudelent prediction process. Please use 'http://<address>/predict' to POST"""

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}
	dt = strftime("[%Y-%b-%d %H:%M:%S]")
	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":

		age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, 
		hours-per-week, native-country = "", "", "", "", "", "", "", "", "", "", "", "", "", ""
		request_json = flask.request.get_json()
		if request_json["age"]:
			description = request_json['age']

		if request_json["workclass"]:
			company_profile = request_json['workclass']

		if request_json["fnlwgt"]:
			benefits = request_json['fnlwgt']
			
		if request_json["education"]:
			benefits = request_json['education']
			
		if request_json["education-num"]:
			benefits = request_json['education-num']
			
		if request_json["marital-status"]:
			benefits = request_json['marital-status']
			
		if request_json["occupation"]:
			benefits = request_json['occupation']
			
		if request_json["relationship"]:
			benefits = request_json['relationship']
			
		if request_json["race"]:
			benefits = request_json['race']
			
		if request_json["sex"]:
			benefits = request_json['sex']
			
		if request_json["capital-gain"]:
			benefits = request_json['capital-gain']
			
		if request_json["capital-loss"]:
			benefits = request_json['capital-loss']
			
		if request_json["hours-per-week"]:
			benefits = request_json['hours-per-week']
			
		if request_json["native-country"]:
			benefits = request_json['native-country']
			
		logger.info(f'{dt} Data: age={age}, workclass={workclass}, fnlwgt={fnlwgt}, education={education}, education-num={education-num}, 
			    marital-status={marital-status}, occupation={occupation}, relationship={relationship}, race={race}, sex={sex}, capital-gain={capital-gain}, 
			    capital-loss={capital-loss}, hours-per-week={hours-per-week}, native-country={native-country}')
		try:
			preds = model.predict(pd.DataFrame({"age": [age], "workclass": [workclass], "fnlwgt": [fnlwgt], "education": [education], 
							    "education-num": [education-num], "marital-status": [marital-status], "occupation": [occupation], 
							    "relationship": [relationship], "race": [race], "sex": [sex], "capital-gain": [capital-gain], 
							    "capital-loss": [capital-loss], "hours-per-week": [hours-per-week], "native-country": [native-country]}))
		except AttributeError as e:
			logger.warning(f'{dt} Exception: {str(e)}')
			data['predictions'] = str(e)
			data['success'] = False
			return flask.jsonify(data)

		data["predictions"] = preds
		# indicate that the request was a success
		data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading the model and Flask starting server..."
		"please wait until server has fully started"))
	port = int(os.environ.get('PORT', 8180))
	app.run(host='0.0.0.0', debug=True, port=port)
