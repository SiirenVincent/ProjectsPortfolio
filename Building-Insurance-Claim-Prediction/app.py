import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# intialisation of flask app
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def main():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def home():

    '''for rendering results on HTML GUI'''
    data1 = request.form['building_dimension']
    data2 = request.form['years_occupied']
    data3 = request.form['yearofobservation']
    data4 = request.form['date_of_occupancy']
    data5 = request.form['insured_period']
    data6 = request.form['building_type']
    data7 = request.form['residential']
    data8 = request.form['expiring_soon']
    # data9 = request.form['building_type']
    # data10 = request.form['date_of_occupancy']
    # data11 = request.form['numberofwindows']
    # data12 = request.form['geo_code']
    # int_features = [int(x)for x in request.form.values()]
    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8]])#, data9, data10, data11, data12]])
    pred = model.predict(arr)
    return render_template('after.html',data=pred)
    # final_features = [np.array(int_features)]
    # prediction  = model.predict(final_features)

#     # output = round(prediction[0],2)

#     return render_template('home.html',  prediction_text= 'You prediction is {}'.format(output))
 
# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     '''
#     For direct API calls through request
#     '''
#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

if __name__=="__main__":
    app.run(debug=True)
