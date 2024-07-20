# from main.main import start
#
# from flask import Flask, request, render_template
# import config.args as args
#
# app = Flask(__name__)
#
#
# @app.route('/', methods=['GET', 'POST'])
# def main():
#     output = ""
#     if request.method == "POST":
#         # input = request.form.get("sequence")
#         input = request.form['sequence']
#         print("-------------",input)
#         output = start(input)
#     return render_template("index.html",resultMessage=output)
#
# # app.run(host='0.0.0.0', port=5000, debug=True)
# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, request, render_template
# from main.main import start  # 确保你从正确的路径导入了start函数
#
# app = Flask(__name__)
#
# @app.route('/', methods=['GET', 'POST'])
# def main():
#     output = ""
#     if request.method == "POST":
#         sequence = request.form['sequence']
#         task = request.form['task']
#         print("Sequence:", sequence)
#         print("Task:", task)
#         output = "xdjwsqidwfcdesc"
#         print("---------------")
#     return render_template("index.html", resultMessage=output)
#
# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request, jsonify
from main.main import start
app = Flask(__name__)

@app.route('/', methods=['GET'])
def root():
    return render_template("index.html")

@app.route('/main', methods=['POST'])
def main():
    if request.method == "POST":
        sequence = request.form['sequence']
        task = request.form['task']
        print("Sequence:", sequence)
        print("Task:", task)
        output = start(input, task)
        print("---------------")
        return jsonify({'resultMessage': output})

if __name__ == '__main__':
    app.run(debug=True)
