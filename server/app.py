from flask import Flask, request, jsonify
from server.env import ShoppingEnv

app = Flask(__name__)
env = ShoppingEnv()


@app.route("/reset", methods=["POST"])
def reset():
    return jsonify(env.reset())


@app.route("/step", methods=["POST"])
def step():
    data = request.get_json(force=True)
    return jsonify(env.step(data.get("action", {})))


@app.route("/", methods=["GET"])
def home():
    return "OK"


def main():
    app.run(host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
