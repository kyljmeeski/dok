from flask import Flask


app = Flask(__name__)


@app.route("/")
def main():
    return "hello"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=True)
