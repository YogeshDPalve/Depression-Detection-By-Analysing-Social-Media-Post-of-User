import torch
from model import Sentiment #importing my file
from flask import request # flask to connect with front end
import flask
import os
from flask import Flask, render_template, request

import random


# Tokenizer / model
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

tokenizer = BertTokenizer.from_pretrained("model/")
model = BertForSequenceClassification.from_pretrained("model/")

import os
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import model

from transformers import BertForQuestionAnswering

from transformers import BertTokenizer, BertForSequenceClassification


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("welcome.html", pred="Welcome to Depression Detection")

@app.route("/gologin")
def login():
    return render_template("index.html", pred="Please ask a question!")

@app.route("/predict", methods=["POST"])
def prediction():
    data = [request.form["question"]]

    name = [request.form["name"]]

    answer = model.Sentiment(data[0]) #["tweet"]

    return render_template(
        "predict.html",
        pred=f"{name} ...I think the answer is {answer} !?",
    )


if __name__ == "__main__":

    app.run(debug=True)
