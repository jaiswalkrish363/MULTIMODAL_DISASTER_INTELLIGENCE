import os
import uuid
import json
import base64
from datetime import datetime
from xml.sax.saxutils import escape

import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import requests

from flask import (
    Flask, render_template, request,
    render_template_string, send_from_directory,
    abort, Response, stream_with_context
)

from werkzeug.utils import secure_filename

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)


app = Flask(__name__)

# ==================================
# CONFIGURATION
# ==================================

UPLOAD_FOLDER = "uploads"
REPORT_FOLDER = "reports"
DATASET_PATH = os.path.join("dataset", "disaster_data.csv")

RISK_MODEL_PATH = os.path.join("models", "disaster_risk_model.pkl")
ENCODER_PATH = os.path.join("models", "severity_encoder.pkl")

IMAGE_MODEL_PATH = os.path.join("models", "disaster_image_model.keras")
IMAGE_CLASSES_PATH = os.path.join("models", "image_classes.json")

OLLAMA_URL = "http://localhost:11434/api/chat"
VISION_MODEL = "qwen2.5vl:3b"

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)


# ==================================
# PDF REPORT GENERATION
# ==================================

def generate_pdf_report(report_type, details):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = (
        f"disaster_report_{timestamp}_{uuid.uuid4().hex[:6]}.pdf"
    )

    filepath = os.path.join(REPORT_FOLDER, filename)

    document = SimpleDocTemplate(
        filepath,
        pagesize=A4,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        textColor=colors.darkblue,
        fontSize=18,
        spaceAfter=18
    )

    content = [
        Paragraph(
            "MULTIMODAL DISASTER INTELLIGENCE PLATFORM",
            title_style
        ),
        Paragraph("Disaster Assessment Report", styles["Heading2"]),
        Spacer(1, 12),
        Paragraph(
            f"Report Type: {escape(str(report_type))}",
            styles["Normal"]
        ),
        Paragraph(
            f"Generated On: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
            styles["Normal"]
        ),
        Spacer(1, 20)
    ]

    table_data = [
        [
            Paragraph("<b>Parameter</b>", styles["Normal"]),
            Paragraph("<b>Result</b>", styles["Normal"])
        ]
    ]

    for key, value in details.items():

        table_data.append([
            Paragraph(escape(str(key)), styles["Normal"]),
            Paragraph(
                escape(str(value)).replace("\n", "<br/>"),
                styles["Normal"]
            )
        ])

    table = Table(
        table_data,
        colWidths=[2.2 * inch, 3.8 * inch]
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP")
    ]))

    content.append(table)
    content.append(Spacer(1, 25))

    content.append(
        Paragraph(
            "Note: This report contains automated image classification, "
            "AI-generated descriptions or historical risk assessment results. "
            "It is not an official emergency warning or a live disaster forecast.",
            styles["Italic"]
        )
    )

    document.build(content)

    return filename


# ==================================
# LOAD IMAGE MODEL
# ==================================

image_model = None
image_classes = None


def load_image_model():

    global image_model, image_classes

    if image_model is None:

        image_model = tf.keras.models.load_model(
            IMAGE_MODEL_PATH
        )

        with open(IMAGE_CLASSES_PATH, "r") as file:
            image_classes = json.load(file)

    return image_model, image_classes


# ==================================
# IMAGE CLASSIFICATION
# ==================================

def classify_disaster_image(filepath):

    model, classes = load_image_model()

    image = tf.keras.utils.load_img(
        filepath,
        target_size=(160, 160)
    )

    image_array = tf.keras.utils.img_to_array(image)

    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(predictions))

    predicted_class = classes[predicted_index]

    confidence = float(predictions[predicted_index]) * 100

    return predicted_class, round(confidence, 2)


# ==================================
# AUTOMATIC FALLBACK DESCRIPTION
# ==================================

def generate_fallback_description(predicted_class):

    descriptions = {

        "flood": (
            "IMAGE DESCRIPTION:\n"
            "The uploaded photograph has been classified as a possible "
            "flood scene. It may represent an area affected by rising "
            "water levels or waterlogging.\n\n"

            "VISIBLE DAMAGE AND OBJECTS:\n"
            "Flood conditions can affect roads, buildings, vehicles "
            "and surrounding infrastructure. The image classifier "
            "does not independently verify individual objects or damage.\n\n"

            "POSSIBLE SITUATION:\n"
            "The scene may represent flooding that could disrupt "
            "transportation and normal activities in the affected area.\n\n"

            "GENERAL SAFETY RECOMMENDATIONS:\n"
            "Avoid walking or driving through floodwater. Stay away "
            "from electrical hazards and follow local emergency instructions."
        ),

        "fire": (
            "IMAGE DESCRIPTION:\n"
            "The uploaded photograph has been classified as a possible "
            "fire scene.\n\n"

            "VISIBLE DAMAGE AND OBJECTS:\n"
            "Fire incidents may affect nearby buildings, vegetation, "
            "vehicles and other surrounding objects. Actual damage "
            "cannot be verified by the category classifier alone.\n\n"

            "POSSIBLE SITUATION:\n"
            "The scene may represent a fire-related incident that "
            "could create heat and smoke hazards.\n\n"

            "GENERAL SAFETY RECOMMENDATIONS:\n"
            "Maintain a safe distance, avoid inhaling smoke and contact "
            "emergency services when necessary."
        ),

        "earthquake": (
            "IMAGE DESCRIPTION:\n"
            "The uploaded photograph has been classified as a possible "
            "earthquake-related scene.\n\n"

            "VISIBLE DAMAGE AND OBJECTS:\n"
            "Earthquakes may cause damage to buildings, roads and "
            "other infrastructure. Specific structural damage "
            "cannot be confirmed by the category classifier alone.\n\n"

            "POSSIBLE SITUATION:\n"
            "The scene may represent an area affected by seismic activity.\n\n"

            "GENERAL SAFETY RECOMMENDATIONS:\n"
            "Stay away from damaged structures and falling debris. "
            "Follow official earthquake safety instructions."
        ),

        "landslide": (
            "IMAGE DESCRIPTION:\n"
            "The uploaded photograph has been classified as a possible "
            "landslide scene.\n\n"

            "VISIBLE DAMAGE AND OBJECTS:\n"
            "Landslides may involve soil, rocks and debris moving down "
            "slopes and can affect nearby roads and buildings.\n\n"

            "POSSIBLE SITUATION:\n"
            "The scene may represent an area affected by unstable "
            "ground or slope movement.\n\n"

            "GENERAL SAFETY RECOMMENDATIONS:\n"
            "Stay away from unstable slopes and affected roads. "
            "Follow local warnings and evacuation instructions."
        ),

        "smoke": (
            "IMAGE DESCRIPTION:\n"
            "The uploaded photograph has been classified as a possible "
            "smoke-related scene.\n\n"

            "VISIBLE DAMAGE AND OBJECTS:\n"
            "Smoke may reduce visibility and can be associated with "
            "fire or other incidents. Its exact source cannot be "
            "confirmed by the category classifier alone.\n\n"

            "POSSIBLE SITUATION:\n"
            "The scene may represent an area affected by smoke.\n\n"

            "GENERAL SAFETY RECOMMENDATIONS:\n"
            "Avoid smoke exposure, move to a safer location if needed "
            "and follow local safety instructions."
        ),

        "normal": (
            "IMAGE DESCRIPTION:\n"
            "The image classifier has identified this photograph "
            "as a normal scene.\n\n"

            "VISIBLE DAMAGE AND OBJECTS:\n"
            "The classifier has not identified a clear disaster category. "
            "It does not independently verify every object or condition.\n\n"

            "POSSIBLE SITUATION:\n"
            "The photograph may represent a scene without an obvious "
            "disaster-related visual category.\n\n"

            "GENERAL SAFETY RECOMMENDATIONS:\n"
            "Continue following normal safety precautions. "
            "A normal classification does not guarantee that an area "
            "is free from danger."
        )
    }

    return descriptions.get(
        str(predicted_class).lower(),
        (
            "IMAGE DESCRIPTION:\n"
            "The uploaded photograph has been processed by the disaster "
            "classification model.\n\n"
            "POSSIBLE SITUATION:\n"
            "The classifier could not provide a detailed category-based "
            "description.\n\n"
            "GENERAL SAFETY RECOMMENDATIONS:\n"
            "Please verify the situation manually and follow official "
            "safety instructions."
        )
    )


# ==================================
# LOCAL AI IMAGE DESCRIPTION STREAM
# ==================================

def stream_disaster_description(filepath):

    with open(filepath, "rb") as image_file:

        image_base64 = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    prompt = """
You are an AI assistant for an academic disaster intelligence project.

Analyze the actual uploaded photograph.

Provide four short sections:

1. IMAGE DESCRIPTION:
Describe what is visibly happening.

2. VISIBLE DAMAGE AND OBJECTS:
Identify visible damage to roads, buildings, vehicles,
trees and other objects.

3. POSSIBLE SITUATION:
Explain what the image appears to show.
Mention uncertainty when the cause is unclear.

4. GENERAL SAFETY RECOMMENDATIONS:
Give basic safety precautions relevant to the image.

Important:
- Analyze the actual photograph.
- Do not invent casualties, locations or measurements.
- Do not claim a building is structurally safe or unsafe.
- Separate visible observations from possibilities.
- Keep the answer clear, concise and useful.
- This is educational AI analysis, not an official emergency assessment.
"""

    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_base64]
            }
        ],
        "stream": True,
        "options": {
            "temperature": 0.2
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        stream=True,
        timeout=(5, 300)
    )

    response.raise_for_status()

    try:

        for line in response.iter_lines():

            if not line:
                continue

            data = json.loads(line)

            message = data.get("message", {})

            text_chunk = message.get("content", "")

            if text_chunk:

                yield {
                    "type": "text",
                    "content": text_chunk
                }

            if data.get("done", False):
                break

    finally:

        response.close()


# ==================================
# DASHBOARD ANALYTICS
# ==================================

def get_dashboard_analytics():

    data = pd.read_csv(DATASET_PATH)

    risk_counts = (
        data["severity"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
    )

    return {
        "total_districts": len(data),
        "low_risk": int(risk_counts["Low"]),
        "medium_risk": int(risk_counts["Medium"]),
        "high_risk": int(risk_counts["High"]),
        "risk_labels": ["Low", "Medium", "High"],
        "risk_values": [
            int(risk_counts["Low"]),
            int(risk_counts["Medium"]),
            int(risk_counts["High"])
        ]
    }


# ==================================
# LOAD NUMERICAL RISK MODEL
# ==================================

def load_risk_model():

    model = joblib.load(RISK_MODEL_PATH)

    encoder = joblib.load(ENCODER_PATH)

    return model, encoder


# ==================================
# HOME PAGE
# ==================================

@app.route("/")
def home():

    analytics = get_dashboard_analytics()

    return render_template(
        "dashboard.html",
        analytics=analytics
    )


# ==================================
# UPLOADED IMAGE
# ==================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# ==================================
# PDF DOWNLOAD
# ==================================

@app.route("/reports/<filename>")
def download_report(filename):

    filename = secure_filename(filename)

    if not filename.endswith(".pdf"):
        abort(404)

    return send_from_directory(
        REPORT_FOLDER,
        filename,
        as_attachment=True
    )


# ==================================
# IMAGE UPLOAD
# ==================================

@app.route("/upload", methods=["POST"])
def upload():

    if "image" not in request.files:
        return "Please select an image.", 400

    file = request.files["image"]

    if file.filename == "":
        return "Please select an image.", 400

    filename = secure_filename(file.filename)

    if not filename or "." not in filename:
        return "Please select a valid image.", 400

    extension = filename.rsplit(".", 1)[-1].lower()

    if extension not in ALLOWED_EXTENSIONS:
        return "Please upload a JPG, JPEG or PNG image.", 400

    unique_filename = str(uuid.uuid4()) + "." + extension

    filepath = os.path.join(
        UPLOAD_FOLDER,
        unique_filename
    )

    try:

        file.save(filepath)

        predicted_class, confidence = classify_disaster_image(filepath)

        return render_template_string("""
        <!DOCTYPE html>
        <html>

        <head>

            <title>Multimodal Disaster Intelligence</title>

            <style>

                body {
                    font-family: Arial, sans-serif;
                    background: #f1f5f9;
                    padding: 30px 15px;
                }

                .result-box {
                    background: white;
                    max-width: 750px;
                    margin: auto;
                    padding: 30px;
                    border-radius: 15px;
                }

                img {
                    max-width: 100%;
                    max-height: 350px;
                    display: block;
                    margin: 20px auto;
                    border-radius: 12px;
                }

                .prediction {
                    font-size: 28px;
                    font-weight: bold;
                    color: #2563eb;
                    text-transform: capitalize;
                }

                .section {
                    background: #f8fafc;
                    padding: 20px;
                    border-radius: 10px;
                    margin-top: 20px;
                    text-align: left;
                    line-height: 1.7;
                    white-space: pre-wrap;
                    overflow-wrap: anywhere;
                    min-height: 100px;
                }

                .status {
                    color: #2563eb;
                    font-weight: bold;
                }

                .note {
                    color: #64748b;
                    line-height: 1.6;
                }

                a {
                    display: inline-block;
                    margin: 12px 6px;
                    padding: 12px 20px;
                    background: #2563eb;
                    color: white;
                    text-decoration: none;
                    border-radius: 8px;
                }

            </style>

        </head>

        <body>

            <div class="result-box">

                <h1>Multimodal AI Disaster Assessment</h1>

                <img
                    src="/uploads/{{ filename }}"
                    alt="Uploaded disaster photograph"
                >

                <h2>Image Classification</h2>

                <p>Predicted category:</p>

                <div class="prediction">
                    {{ predicted_class }}
                </div>

                <p>
                    Model confidence: {{ confidence }}%
                </p>

                <hr>

                <h2>Image Description</h2>

                <p class="status" id="status">
                    Generating image description...
                </p>

                <div class="section" id="description">
                    Please wait...
                </div>

                <p class="note">
                    Descriptions are generated using local vision AI when
                    available, or an automatic category-based description.
                    Results may be inaccurate and should be verified.
                </p>

                <div id="report-area"></div>

                <a href="/">
                    Return to Dashboard
                </a>

            </div>

            <script>

                async function startAIAnalysis() {

                    const status = document.getElementById("status");

                    const output = document.getElementById("description");

                    const reportArea = document.getElementById("report-area");

                    output.textContent = "";

                    try {

                        const response = await fetch(
                            "/describe/{{ filename }}"
                        );

                        if (!response.ok || !response.body) {
                            throw new Error("Description unavailable");
                        }

                        const reader = response.body.getReader();

                        const decoder = new TextDecoder();

                        let buffer = "";

                        while (true) {

                            const result = await reader.read();

                            if (result.done) {
                                break;
                            }

                            buffer += decoder.decode(
                                result.value,
                                { stream: true }
                            );

                            let boundary;

                            while ((boundary = buffer.indexOf("\\n\\n")) !== -1) {

                                const event = buffer.substring(0, boundary);

                                buffer = buffer.substring(boundary + 2);

                                const dataLine = event
                                    .split("\\n")
                                    .find(line => line.startsWith("data: "));

                                if (!dataLine) continue;

                                const data = JSON.parse(
                                    dataLine.substring(6)
                                );

                                if (data.type === "reset") {

                                    output.textContent = "";

                                    status.textContent =
                                        "Using automatic image description...";

                                }

                                if (data.type === "text") {

                                    output.textContent += data.content;

                                }

                                if (data.type === "done") {

                                    status.textContent = "Description ready.";

                                    if (data.report_filename) {

                                        const link = document.createElement("a");

                                        link.href = "/reports/" +
                                            encodeURIComponent(data.report_filename);

                                        link.textContent = "Download PDF Report";

                                        reportArea.appendChild(link);
                                    }

                                }

                            }

                        }

                        if (!output.textContent.trim()) {

                            output.textContent =
                                "No description could be generated. Please try another image.";

                            status.textContent = "Description unavailable.";

                        }

                    } catch (error) {

                        status.textContent = "Description unavailable.";

                        output.textContent =
                            "The description could not be generated. Please try uploading another image.";

                    }

                }

                startAIAnalysis();

            </script>

        </body>

        </html>
        """,
        filename=unique_filename,
        predicted_class=predicted_class,
        confidence=confidence)

    except Exception:

        if os.path.exists(filepath):
            os.remove(filepath)

        return "Unable to process this image. Please try another image.", 500


# ==================================
# STREAM DESCRIPTION WITH FALLBACK
# ==================================

@app.route("/describe/<filename>")
def describe_image(filename):

    filename = secure_filename(filename)

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    if not filename or not os.path.isfile(filepath):
        abort(404)

    def generate():

        full_description = ""
        source = "Category-based fallback"

        # Try Ollama first
        try:

            for item in stream_disaster_description(filepath):

                if item["type"] == "text":

                    full_description += item["content"]

                    yield (
                        "data: "
                        + json.dumps(item)
                        + "\n\n"
                    )

            if not full_description.strip():

                raise RuntimeError("Empty AI response")

            source = "Qwen2.5-VL"

        except Exception:

            # Clear partially generated Ollama text
            full_description = ""

            yield (
                "data: "
                + json.dumps({
                    "type": "reset"
                })
                + "\n\n"
            )

            # Automatically use image classification fallback
            try:

                predicted_class, confidence = (
                    classify_disaster_image(filepath)
                )

                full_description = generate_fallback_description(
                    predicted_class
                )

                source = "Category-based fallback"

            except Exception:

                full_description = (
                    "The uploaded photograph was received, but a "
                    "description could not be generated. Please try again."
                )

                source = "Basic fallback"

            # Stream fallback text
            for i in range(0, len(full_description), 12):

                chunk = full_description[i:i + 12]

                yield (
                    "data: "
                    + json.dumps({
                        "type": "text",
                        "content": chunk
                    })
                    + "\n\n"
                )

        # Generate PDF report
        report_filename = ""

        try:

            report_filename = generate_pdf_report(
                "Disaster Image Assessment",
                {
                    "Uploaded Image": filename,
                    "Description Source": source,
                    "Image Description": full_description
                }
            )

        except Exception:

            pass

        yield (
            "data: "
            + json.dumps({
                "type": "done",
                "report_filename": report_filename
            })
            + "\n\n"
        )

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


# ==================================
# NUMERICAL DISASTER RISK PREDICTION
# ==================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        human_fatality = float(request.form["human_fatality"])

        human_injured = float(request.form["human_injured"])

        population = float(request.form["population"])

        mean_flood_duration = float(
            request.form["mean_flood_duration"]
        )

        flooded_area = float(
            request.form["corrected_percent_flooded_area"]
        )

        values = [
            human_fatality,
            human_injured,
            population,
            mean_flood_duration,
            flooded_area
        ]

        if any(value < 0 for value in values):
            return "Values cannot be negative.", 400

        model, encoder = load_risk_model()

        input_data = pd.DataFrame([{
            "Human_fatality": human_fatality,
            "Human_injured": human_injured,
            "Population": population,
            "Mean_Flood_Durationt": mean_flood_duration,
            "Corrected_Percent_Flooded_Area": flooded_area
        }])

        prediction = model.predict(input_data)[0]

        severity = encoder.inverse_transform([prediction])[0]

        probabilities = model.predict_proba(input_data)[0]

        confidence = round(
            float(max(probabilities)) * 100,
            2
        )

        report_filename = generate_pdf_report(
            "Historical Flood Risk Assessment",
            {
                "Human Fatalities": human_fatality,
                "Human Injured": human_injured,
                "Population": population,
                "Mean Flood Duration": mean_flood_duration,
                "Corrected Percent Flooded Area": flooded_area,
                "Predicted Risk Category": severity,
                "Model Confidence": f"{confidence}%"
            }
        )

        return render_template_string("""
        <!DOCTYPE html>

        <html>

        <head>

            <title>Disaster Risk Result</title>

            <style>

                body {
                    font-family: Arial;
                    background: #f1f5f9;
                    text-align: center;
                    padding: 40px 15px;
                }

                .result-box {
                    background: white;
                    max-width: 500px;
                    margin: auto;
                    padding: 30px;
                    border-radius: 15px;
                }

                .severity {
                    font-size: 30px;
                    font-weight: bold;
                }

                .Low { color: green; }
                .Medium { color: orange; }
                .High { color: red; }

                a {
                    display: inline-block;
                    margin: 12px 6px;
                    padding: 12px 20px;
                    background: #2563eb;
                    color: white;
                    text-decoration: none;
                    border-radius: 8px;
                }

            </style>

        </head>

        <body>

            <div class="result-box">

                <h1>Disaster Risk Assessment</h1>

                <p>Predicted historical risk category:</p>

                <div class="severity {{ severity }}">
                    {{ severity }} Risk
                </div>

                <p>Model confidence: {{ confidence }}%</p>

                <p>
                    Classification based on historical flood-impact
                    indicators and DFSI-derived labels.
                    This is not a live flood forecast.
                </p>

                <a href="/reports/{{ report_filename }}">
                    Download PDF Report
                </a>

                <a href="/">
                    Return to Dashboard
                </a>

            </div>

        </body>

        </html>
        """,
        severity=severity,
        confidence=confidence,
        report_filename=report_filename)

    except ValueError:

        return "Please enter valid numeric values.", 400

    except Exception:

        return "Unable to complete the risk prediction.", 500


# ==================================
# ERROR HANDLING
# ==================================

@app.errorhandler(413)
def file_too_large(error):

    return (
        "Image is too large. Maximum upload size is 10 MB.",
        413
    )


# ==================================
# RUN APPLICATION
# ==================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )