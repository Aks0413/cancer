import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

#Flask App
app = Flask(__name__, template_folder="Template")

#Loading Model
model = pickle.load(open('server_flask\model.pkl' , 'rb'))


@app.route('/', methods=["GET"])
def home_page():
    return render_template("homepage.html")

@app.route('/index1', methods=["POST"])
def first_page():
    return render_template("index.html")

@app.route('/index2', methods = ["POST"])
def second_page():
    return render_template("index2.html")


@app.route('/index3', methods = ["POST"])
def third_page():
    return render_template("index3.html")

@app.route('/index4', methods = ["POST"])
def fourth_page():
    return render_template("index4.html")

@app.route('/index5', methods = ["POST"])
def fifth_page():
    return render_template("index5.html")

@app.route('/index6', methods = ["POST"])
def sixth_page():
    return render_template("index6.html")

@app.route('/index7', methods = ["POST"])
def seventh_page():
    return render_template("index7.html")

@app.route('/index8', methods = ["POST"])
def eigth_page():
    return render_template("index8.html")

@app.route('/index9', methods = ["POST"])
def ninth_page():
    return render_template("index9.html")

@app.route('/index10', methods = ["POST"])
def tenth_page():
    return render_template("index10.html")

@app.route('/index11', methods = ["POST"])
def eleventh_page():
    return render_template("index11.html")

@app.route('/index12', methods = ["POST"])
def twelth_page():
    return render_template("index12.html")

@app.route('/index13', methods = ["POST"])
def thirteenth_page():
    return render_template("index13.html")

@app.route('/index14', methods = ["POST"])
def fourteenth_page():
    return render_template("index14.html")

@app.route('/index15', methods = ["POST"])
def fithteenth_page():
    return render_template("index15.html")

@app.route('/index16', methods = ["POST"])
def sixteenth_page():
    return render_template("index16.html")

@app.route('/index17', methods = ["POST"])
def seventeenth_page():
    return render_template("index17.html")

@app.route('/index18', methods = ["POST"])
def eighteen_page():
    return render_template("index18.html")

@app.route('/index19', methods = ["POST"])
def nineteenth_page():
    return render_template("index19.html")

@app.route('/index20', methods = ["POST"])
def twentieth_page():
    return render_template("index20.html")

@app.route('/index21', methods = ["POST"])
def twentyfirst_page():
    return render_template("index21.html")

@app.route('/index22', methods = ["POST"])
def twentysecond_page():
    return render_template("index22.html")

@app.route('/index23', methods = ["POST"])
def twentythird_page():
    return render_template("index23.html")



@app.route('/predict', methods = ["POST"])
def predict():
    print(request.form)
    input1 = float(request.form.get("1", 0.0))
    input2 = float(request.form.get("2", 0.0))
    input3 = float(request.form.get("3", 0.0))
    input4 = float(request.form.get("4", 0.0))
    input5 = float(request.form.get("5", 0.0))
    input6 = float(request.form.get("6", 0.0))
    input7 = float(request.form.get("7", 0.0))
    input8 = float(request.form.get("8", 0.0))
    input9 = float(request.form.get("9", 0.0))
    input10 = float(request.form.get("10", 0.0))
    input11 = float(request.form.get("11", 0.0))
    input12 = float(request.form.get("12", 0.0))
    input13 = float(request.form.get("13", 0.0))
    input14 = float(request.form.get("14", 0.0))
    input15 = float(request.form.get("15", 0.0))
    input16 = float(request.form.get("16", 0.0))
    input17 = float(request.form.get("17", 0.0))
    input18 = float(request.form.get("18", 0.0))
    input19 = float(request.form.get("19", 0.0))
    input20 = float(request.form.get("20", 0.0))
    input21 = float(request.form.get("21", 0.0))
    input22 = float(request.form.get("22", 0.0))
    input23 = float(request.form.get("23", 0.0))


    # Print input values for debugging
    print("Input Values:", input1, input2, input3, input4, input5, input6, input7, input23)
    features = torch.tensor([[input1, input2, input3, input4, input5, input6, input7, input8, input9, input10, input11, input12, input13, input14, input15, input16, input17, input18, input19, input20, input21, input22, input23]])

    prediction = model(features)
    
    # Print intermediate values for debugging
    print("Features:", features)
    print("Prediction:", prediction)

    Probabilities = nn.functional.softmax(prediction, dim=1)
    final_prediction = torch.argmax(Probabilities, dim=1)

    # Print final values for debugging
    print("Probabilities:", Probabilities)
    print("Final Prediction:", final_prediction)

    text_pred = "Low Chance"

    if final_prediction.item() == 1:
        text_pred = "Medium Chance"
    elif final_prediction.item() == 2:
        text_pred = "High Chance"

    # Print the predicted values for debugging
    print("Final Text Prediction:", text_pred)
    print("Final Numeric Prediction:", final_prediction.item())

    return render_template("predict.html", prediction_text=text_pred, value_text=final_prediction.item())

if __name__ == "__main__":
    app.run(debug=True)
