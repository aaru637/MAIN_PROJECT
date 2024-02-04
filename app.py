from flask import Flask, request, render_template
from Fake_Instagram_Detection_Using_ANN import *

app = Flask(__name__)


def setter(data):
    return [
        int(data['profile_pic']),
        float(find_ratio_of_name(data['username'])),
        int(find_no_of_words_in_name(data['fullname'])),
        float(find_ratio_of_name(data['fullname'])),
        int(name_is_equal_to_username(data['fullname'], data['username'])),
        int(len(data['description'])),
        int(data['external_url']),
        int(data['private']),
        int(data['post']),
        int(data['followers']),
        int(data['follows'])
    ]


@app.route('/', )
def hello_world():  # put application's code here
    return render_template('index.html')


@app.route('/evaluate', methods=['POST'])
def evaluate():
    if request.method == 'POST':
        data = request.form
        return compute(setter(data))


if __name__ == '__main__':
    app.run()
