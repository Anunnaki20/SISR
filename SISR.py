# Web Framework
from flask import Flask
from flask import request
from flask import jsonify

# Create the Flask Web application
app = Flask(__name__)

# Base URL
@app.route('/', methods=["POST"])
def test():
    data = request.get_json()
    print(data)
    return jsonify(f"Hey! {data}")


# Run the server on the local host
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')