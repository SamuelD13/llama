import flask
from transmitter import Transmitter
import subprocess
import time

app = flask.Flask(__name__)
transmitter = Transmitter()

@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/prompt', methods=['POST'])
def prompt() :
    if flask.request.method == 'POST':
        question = flask.request.form['user_input']

        with open("input_txt", 'w') as file :
            file.write(question)

        try :
            result = subprocess.run(['sbatch', "transmit.sbatch"], check=True, stdout=subprocess.PIPE, text=True)
            id = result.stdout.split()[-1]
            print("Sbatch script executed successfully. Job ID :", id)
        except subprocess.CalledProcessError as e:
            print(f"Error executing sbatch script: {e}")

        while True :
            try :
                status = subprocess.check_output(['squeue', '-j', id], text=True)
                print("Job status:", status)
                if id not in status.split() :
                    print("Job is done")
                    break
            except subprocess.CalledProcessError :
                print("Error searching for the job")
                break
            time.sleep(10)

        with open("output/answer.txt", 'r') as file:
            answer = file.read()

        return flask.render_template('index.html', output=answer)

if __name__ == '__main__':
    app.run(debug=True)