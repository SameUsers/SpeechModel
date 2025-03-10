from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from main import process_call

app = Flask(__name__)
CORS(app)  # Разрешаем CORS для работы с фронтом

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Создаем папку для файлов
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Главная страница (рендерим HTML)
@app.route("/")
def index():
    return render_template("index.html")


# API для обработки загруженных файлов
@app.route("/analyze", methods=["POST"])
def analyze_call():
    if "file" not in request.files:
        return jsonify({"error": "Файл не найден"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Файл не выбран"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Анализируем звонок
    result = process_call(file_path)

    return jsonify({"status": "success", "result": result})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)