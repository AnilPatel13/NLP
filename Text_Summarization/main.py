from flask import Flask, render_template, request, redirect, url_for, session, flash
from module.bert_model import bert_model
from module.t5_model import t5_model
from module.gp2_model import gp2_model
from module.xlnet_model import xlnet_model
from module.website_text_summarization import website_text_summarization
from module.user_query_summarization import user_query_summarization
from module.hybrid_model import hybrid_model
from module.pegasus import run_pegasus
from helper.Constant import *
import os

app = Flask(__name__)

app.secret_key = 'Tiger123'


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html")


@app.route('/')
@app.route('/home')
def home():
    if request.method == "GET":
        return render_template("index.html")
    else:
        return render_template("404.html")


@app.route('/bert', methods=["POST", "GET"])
def bert():
    if request.method == "POST":

        # Form
        actual_text = request.form.get("inputText")

        if actual_text in ("", None):
            flash("Please Provide input Paragraph", "danger")
            return render_template("bert.html")

        text_summarized = bert_model(actual_text)

        if actual_text in ("", None):
            flash("Unable to Perform Text Summarization | Retry Again !", "danger")
            return render_template("bert.html")
        else:
            flash("Sucessfully Run Bert Pre-Trained Model", "success")

        return render_template("bert.html", textSummarized=text_summarized, output=PROJECT.OUTPUT)
    elif request.method == "GET":
        return render_template("bert.html")
    else:
        return render_template("404.html")


@app.route('/t5_Nlp_model', methods=["POST", "GET"])
def t5_Nlp_model():
    if request.method == "POST":

        # Form
        actual_text = request.form.get("inputText")

        if actual_text in ("", None):
            flash("Please Provide input Paragraph", "danger")
            return render_template("t5_model.html")

        text_summarized = t5_model(actual_text)

        if actual_text in ("", None):
            flash("Unable to Perform Text Summarization | Retry Again !", "danger")
            return render_template("t5_model.html")
        else:
            flash("Sucessfully Run T5 Pre-Trained Model", "success")

        return render_template("t5_model.html", textSummarized=text_summarized, output=PROJECT.OUTPUT)
    elif request.method == "GET":
        return render_template("t5_model.html")
    else:
        return render_template("404.html")


@app.route('/gpt2_Nlp_model', methods=["POST", "GET"])
def gpt2_Nlp_model():
    if request.method == "POST":

        # Form
        actual_text = request.form.get("inputText")

        if actual_text in ("", None):
            flash("Please Provide input Paragraph", "danger")
            return render_template("gpt2_model.html")

        text_summarized = gp2_model(actual_text)

        if actual_text in ("", None):
            flash("Unable to Perform Text Summarization | Retry Again !", "danger")
            return render_template("gpt2_model.html")
        else:
            flash("Sucessfully Run GPT2 Pre-Trained Model", "success")

        return render_template("gpt2_model.html", textSummarized=text_summarized, output=PROJECT.OUTPUT)
    elif request.method == "GET":
        return render_template("gpt2_model.html")
    else:
        return render_template("404.html")


@app.route('/xlnet_Nlp_model', methods=["POST", "GET"])
def xlnet_Nlp_model():
    if request.method == "POST":

        # Form
        actual_text = request.form.get("inputText")

        if actual_text in ("", None):
            flash("Please Provide input Paragraph", "danger")
            return render_template("xlnet_model.html")

        text_summarized = xlnet_model(actual_text)

        if actual_text in ("", None):
            flash("Unable to Perform Text Summarization | Retry Again !", "danger")
            return render_template("xlnet_model.html")
        else:
            flash("Sucessfully Run XLNET Pre-Trained Model", "success")

        return render_template("xlnet_model.html", textSummarized=text_summarized, output=PROJECT.OUTPUT)
    elif request.method == "GET":
        return render_template("xlnet_model.html")
    else:
        return render_template("404.html")


@app.route('/website_Nlp_model', methods=["POST", "GET"])
def website_Nlp_model():
    if request.method == "POST":

        # Form
        actual_text = request.form.get("inputText")

        if actual_text in ("", None):
            flash("Please Provide input URL Correctly", "danger")
            return render_template("web_page.html")

        text_summarized = website_text_summarization(actual_text)

        if actual_text in ("", None):
            flash("Unable to Perform Text Summarization | Retry Again !", "danger")
            return render_template("web_page.html")
        else:
            flash("Sucessfully Run Summary", "success")

        return render_template("web_page.html", textSummarized=text_summarized, output=PROJECT.OUTPUT)
    elif request.method == "GET":
        return render_template("web_page.html")
    else:
        return render_template("404.html")


@app.route('/user_query', methods=["POST", "GET"])
def user_query():
    if request.method == "POST":

        # Form
        actual_text = request.form.get("inputText")

        if actual_text in ("", None):
            flash("Please Provide input Paragraph", "danger")
            return render_template("user_query.html")

        text_summarized = user_query_summarization(actual_text)

        if actual_text in ("", None):
            flash("Unable to Perform Text Summarization | Retry Again !", "danger")
            return render_template("user_query.html")
        else:
            flash("Sucessfully Run Summary", "success")

        return render_template("user_query.html", textSummarized=text_summarized, output=PROJECT.OUTPUT)
    elif request.method == "GET":
        return render_template("user_query.html")
    else:
        return render_template("404.html")


@app.route('/pegasus', methods=["POST", "GET"])
def pegasus():
    if request.method == "POST":

        # Form
        actual_text = request.form.get("inputText")

        if actual_text in ("", None):
            flash("Please Provide input Paragraph", "danger")
            return render_template("pegasus.html")

        text_summarized = run_pegasus(actual_text)

        if actual_text in ("", None):
            flash("Unable to Perform Text Summarization | Retry Again !", "danger")
            return render_template("pegasus.html")
        else:
            flash("Sucessfully Run Pegasus Model", "success")

        return render_template("pegasus.html", textSummarized=text_summarized, output=PROJECT.OUTPUT)
    elif request.method == "GET":
        return render_template("pegasus.html")
    else:
        return render_template("404.html")


@app.route('/hybrid', methods=["POST", "GET"])
def hybrid():
    if request.method == "POST":

        # Form
        actual_text = request.form.get("inputText")

        if actual_text in ("", None):
            flash("Please Provide input Paragraph", "danger")
            return render_template("hybrid.html")

        text_summarized = hybrid_model(actual_text)

        if actual_text in ("", None):
            flash("Unable to Perform Text Summarization | Retry Again !", "danger")
            return render_template("hybrid.html")
        else:
            flash("Sucessfully Run Hybrid Model", "success")

        return render_template("hybrid.html", textSummarized=text_summarized, output=PROJECT.OUTPUT)
    elif request.method == "GET":
        return render_template("hybrid.html")
    else:
        return render_template("404.html")


if __name__ == "__main__":
    # app.run(debug=True, port=int(os.environ.get("PORT", 8080)))
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
