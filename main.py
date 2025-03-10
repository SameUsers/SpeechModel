import whisper
import google.generativeai as genai
import os
import librosa
import noisereduce as nr
import soundfile as sf
import subprocess

# 🔹 Указываем путь к ffmpeg (если нужно)
ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)

# 🔹 Настраиваем API Gemini
genai.configure(api_key="AIzaSyArCJQTAbh1lddE1svdp3iDlKnefpdBols")

# 🔹 Загружаем модель Whisper
print("Загружаем модель Whisper...")
whisper_model = whisper.load_model("base")  # Можно "small" или "large"


# 🔹 Функция для конвертации аудио в WAV
def convert_audio_to_wav(input_file):
    output_file = "converted_audio.wav"
    cmd = [
        ffmpeg_path, "-y", "-i", input_file,
        "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
        output_file
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Аудио {input_file} сконвертировано в {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при конвертации: {e}")
        return input_file  # Если ошибка, работаем с оригиналом


# 🔹 Функция для шумоподавления
def reduce_noise(audio_file):
    print(f"Шумоподавление для: {audio_file}")
    y, sr = librosa.load(audio_file, sr=16000)
    y_denoised = nr.reduce_noise(y=y, sr=sr)
    cleaned_file = "cleaned_" + audio_file
    sf.write(cleaned_file, y_denoised, sr)
    return cleaned_file


# 🔹 Функция для распознавания речи
def transcribe_audio(file_path):
    wav_file = convert_audio_to_wav(file_path)  # Конвертация
    clean_file = reduce_noise(wav_file)  # Убираем шум
    print(f"Обрабатываем аудио: {clean_file}")
    result = whisper_model.transcribe(clean_file)
    return result["text"]


# 🔹 Функция анализа текста через Gemini
def analyze_text_with_gemini(text):
    prompt = (
        "Ты аналитик диалогов. Проанализируй текст и определи его тон (positive, neutral, negative). "
        "Также оцени профессионализм и вежливость менеджера. Дай короткий комментарий."
        f"\n\nТекст: {text}"
    )

    # Загружаем модель
    model = genai.GenerativeModel("models/gemini-2.0-flash-thinking-exp-1219")

    # Отправляем запрос к модели
    response = model.generate_content([prompt])

    # Проверяем, есть ли текст в ответе
    if hasattr(response, "text"):
        return response.text.strip()
    return "Ошибка при анализе текста."





def calculate_score(text):
    analysis = analyze_text_with_gemini(text)

    # Извлекаем оценку (если модель вернула её явно)
    score = 3  # По умолчанию
    if "positive" in analysis.lower():
        score = 5
    elif "negative" in analysis.lower():
        score = 1

    return score, analysis

# 🔹 Основной процесс
def process_call(audio_file):
    text = transcribe_audio(audio_file)
    print(f"Распознанный текст: {text}")

    score, analysis = calculate_score(text)
    print(f"Анализ от Gemini: {analysis}")
    print(f"Итоговый скор-балл: {score}/5")
    return analysis


# 🔹 Тестируем
if __name__ == "__main__":
    process_call(audio)