import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('svm_model1.pkl', 'rb'))


@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['GET','post'])
def predict():
	
	age = int(request.form['age'])
	sex = int(request.form['sex'])
	cp = int(request.form['cp'])
	trestbps = int(request.form['trestbps'])
	chol = int(request.form['chol'])
	fbs = int(request.form['fbs'])
	restecg = int(request.form['restecg'])
	thalach = int(request.form['thalach'])
	exang = int(request.form['exang'])
	oldpeak = float(request.form['oldpeak'])
	slope = int(request.form['slope'])
	ca = int(request.form['ca'])
	thal = int(request.form['thal'])

	
	final_features = pd.DataFrame([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
	
	predict = model.predict(final_features)
	
	output = predict[0]
	
	if output==0:
		HD="***Absence of Heart Disease***"
	else:
		HD="***Presence of Heart Disease***"
	
	return render_template('index.html', prediction_text='{}'.format(HD))
	
if __name__ == "__main__":
	app.run(debug=True)
