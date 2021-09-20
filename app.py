from flask import Flask,render_template,request,redirect
from model_training.train_model import Predict
from flask_cors import CORS,cross_origin
app = Flask(__name__)

@app.route("/",methods=["GET"])
@cross_origin()
def home_page():
    return render_template("home_page.html")

@app.route("/predict",methods=["GET","POST"])
@cross_origin()
def predict():
    values=[]
    if request.method=="POST":
        model_type=request.form.get('model_type')
        values.append(float(request.form.get('tv')))
        values.append(float(request.form.get('radio')))
        values.append(float(request.form.get('news')))
        return render_template("score_page.html",score=Predict().predict_outcome("Lasso",values))
    else:
        return redirect("/")




if __name__=="__main__":
    app.run()