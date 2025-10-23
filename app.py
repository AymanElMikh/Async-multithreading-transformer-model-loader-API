from flask import Flask, request, jsonify
from inference_service import inference_service 


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    """
    API endpoint to receive text and return the model's prediction.
    """

    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    data = request.get_json()
    input_text = data.get('text')

    if not input_text:
        return jsonify({"error": "Missing 'text' field in JSON data"}), 400

    try:
        prediction = inference_service.predict(input_text)
        return jsonify(prediction), 200
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

@app.route('/')
def health_check():
    return jsonify({"status": "Transformer API is running", "model": inference_service.classifier.model.config.name_or_path})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded= True)