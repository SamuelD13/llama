import flask
from transmitter import Transmitter
import subprocess

app = flask.Flask(__name__)
transmitter = Transmitter()

@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/prompt', methods=['POST'])
def prompt():
    if flask.request.method == 'POST':
        question = flask.request.form['user_input']
        with open("input_txt", 'w') as file:
            file.write(question)
        subprocess.run(['sbatch', "transmit.batch"])
        with open("transmit-output.log", 'r') as file:
            answer = file.read()
        return flask.render_template('index.html', output=answer)

if __name__ == '__main__':
    app.run(debug=True)