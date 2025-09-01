from flask import Flask, request, jsonify
import uvicorn
import joblib
import json
import tempfile
import os
from werkzeug.exceptions import RequestEntityTooLarge, BadRequest
from src.config import Config
from src.model import LegalBertClassifier
from src.helper import extract_text_smart, clean_contract_text, format_snippet, sniff_is_pdf
svm_model = joblib.load("./research/svm_contract_classifier.pkl")
vectorizer = joblib.load("./research/tfidf_vectorizer.pkl")

#load model once setup
classifier = LegalBertClassifier(Config.MODEL_DIR)



app=Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

# Text classification endpoint
@app.route("/predict-text", methods=["POST"])
def predict_text():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    cleaned = clean_contract_text(data["text"])
    features = vectorizer.transform([cleaned])
    prediction = svm_model.predict(features)[0]
    return jsonify({"prediction": prediction})

# PDF classification endpoint
@app.route("/predict-pdf", methods=["POST"])
def predict_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        extracted_text = extract_text_smart(tmp_path)
    finally:
        os.remove(tmp_path)  # cleanup

    if not extracted_text.strip():
        return jsonify({"error": "Could not extract text from PDF"}), 400

    cleaned = clean_contract_text(extracted_text)
    features = vectorizer.transform([cleaned])
    prediction = svm_model.predict(features)[0]

    return jsonify({
        "prediction": prediction,
        "extracted_text_snippet":format_snippet(extracted_text)  # preview only
    })


@app.get("/healthz")
def health():
    return jsonify({"status": "ok"})

@app.get("/readyz")
def ready():
    # Simple readiness check (model + tokenizer loaded)
    return jsonify({
        "status": "ready",
        "model_dir": Config.MODEL_DIR,
        "labels": list(classifier.label2id.keys())
    }) 
@app.post("/v1/predict")
def predict():
    """
    Accepts:
      - JSON: { "text": "..." }
      - multipart/form-data with a PDF file field named "file"
    Returns:
      - prediction, confidence, probabilities, snippet, chunks_used
    """
    try:
        # 1) If text provided in JSON
        if request.is_json:
            data = request.get_json(silent=True) or {}
            text = data.get("text", "")
            if not text or not text.strip():
                raise BadRequest("JSON body must include non-empty 'text'")
            cleaned = clean_contract_text(text)
            label, conf, probs, chunks = classifier.predict(cleaned)
            return jsonify({
                "prediction": label,
               # "confidence": conf,
                "probabilities": probs,
               # "chunks_used": chunks,
                "snippet": format_snippet(text)
            })

        # 2) If a file is provided (multipart/form-data)
        if "file" in request.files:
            f = request.files["file"]
            if not f.filename:
                raise BadRequest("Empty filename")

            # Basic sniffing
            if not sniff_is_pdf(f.filename, f.mimetype):
                raise BadRequest("Only PDF files are supported in 'file'")

            file_bytes = f.read()
            if not file_bytes:
                raise BadRequest("Uploaded file is empty")

            # Save file temporarily for extraction
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            try:
                # Extract → Clean → Predict
                extracted = extract_text_smart(tmp_path)
            finally:
                os.remove(tmp_path)  # cleanup
            if not extracted or not extracted.strip():
                raise BadRequest("Could not extract text from PDF (even with OCR)")

            cleaned = clean_contract_text(extracted)
            label, conf, probs, chunks = classifier.predict(cleaned)
            return jsonify({
                "prediction": label,
                #"confidence": conf,
                "probabilities": probs,
                #"chunks_used": chunks,
                "snippet": format_snippet(extracted)
            })

        raise BadRequest("Send JSON {'text': ...} or multipart/form-data with a 'file' PDF")
    except RequestEntityTooLarge:
        return jsonify({"error": "Payload too large"}), 413
    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Avoid leaking internals in prod; log e in real deployments
        return jsonify({"error": "Internal server error"}), 500


# ======================
# Run Flask
# ======================
if __name__ == "__main__":
    app.run(debug=True, port=8080)