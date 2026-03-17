from flask import Flask
app = Flask(__name__)

@app.route('/') # 这是一个装饰器，用于告诉 Flask 哪个 URL 应该触发下面的函数。
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)