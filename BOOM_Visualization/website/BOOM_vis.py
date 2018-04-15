from flask import Flask, render_template, request, redirect

app = Flask(__name__)

@app.route('/classifier', methods = ['POST'])
def classifier():
    q1 = request.form['q1']
    q2 = request.form['q2']
    #result = classify_duplicates q1 q2
    return render_template('classifier.html', q1 = q1, q2 = q2, result = "")#result)

@app.route('/')
def default():
    return render_template('index.html')

if __name__ == "__main__":
    app.run()