from flask import Flask, render_template, request, redirect
import sys
import os
sys.path.insert(0, "../../nn")
from nn import Model
os.chdir("../../nn")
m = Model(fold_num=0)
m.load_pretrained(model_name='gru_v5_fold0.h5',model_func=m.gru_similarity_model)
os.chdir("../vis_website/website")



app = Flask(__name__)

@app.route('/classifier', methods = ['POST'])
def classifier():
    q1 = request.form['q1']
    q2 = request.form['q2']
    result = m.is_dup(q1, q2)
    result_str = "Duplicates" if round(result) == 1 else "Not Duplicates"
    return render_template('classifier.html', q1 = q1, q2 = q2, result = result, result_string = result_str)

@app.route('/')
def default():
	return render_template('index.html')

if __name__ == "__main__":
    app.run()