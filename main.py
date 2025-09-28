import os
import tempfile
import subprocess
import json
import io
import docx
import time # Cần cho việc chờ kết quả từ Azure
from fastapi import FastAPI, Form, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import azure.cognitiveservices.speech as speechsdk

# --- Thư viện mới cho Azure AI Vision ---
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# Tải các biến môi trường từ file .env.
load_dotenv()

# --- Cấu hình API ---
# Gemini API (cho dịch thuật và tạo câu hỏi)
gemini_api_key = os.getenv("GEMINI_API_KEY")
try:
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)
    else:
        print("Cảnh báo: GEMINI_API_KEY chưa được thiết lập.")
except Exception as e:
    print(f"Lỗi khi cấu hình Gemini API: {e}")

# --- Khởi tạo FastAPI App ---
app = FastAPI()

# --- HÀM TIỆN ÍCH ---

def perform_recognition(wav_path: str, reference_text: str) -> dict:
    try:
        speech_key = os.getenv("SPEECH_KEY")
        speech_region = os.getenv("SPEECH_REGION")

        if not speech_key or not speech_region:
            return {"error": "Server is missing required Azure Speech credentials."}

        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        
        pronunciation_config = speechsdk.PronunciationAssessmentConfig(
            reference_text=reference_text,
            grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
            granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
            enable_miscue=True
        )

        audio_config = speechsdk.audio.AudioConfig(filename=wav_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        pronunciation_config.apply_to(recognizer)
        result = recognizer.recognize_once_async().get()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            if result.text and result.text.strip() and result.text.strip() != ".":
                pron_result = speechsdk.PronunciationAssessmentResult(result)
                return {"text": result.text, "accuracy_score": pron_result.accuracy_score}
            else:
                return {"error": "No speech could be recognized. Please speak clearly."}
        elif result.reason == speechsdk.ResultReason.NoMatch:
            return {"error": "No speech could be recognized. Please speak clearly."}
        elif result.reason == speechsdk.ResultReason.Canceled:
            details = result.cancellation_details
            return {"error": str(details.error_details)}
            
    except Exception as e:
        print(f"An error occurred during recognition: {e}")
        return {"error": "An internal error occurred during speech recognition."}
    
    return {"error": "Unknown recognition error."}

# --- ENDPOINTS IELTS ---

@app.post("/get-ielts-questions")
async def get_ielts_questions(topic: str = Form(...)):
    if not gemini_api_key:
        return JSONResponse(status_code=500, content={"error": "Gemini API key is not configured."})
    
    model_name = 'gemini-pro-latest' 
    try:
        print(f"Attempting to use confirmed available model: '{model_name}'")
        model = genai.GenerativeModel(model_name)
        prompt = f"""
        Generate a list of 4 common IELTS Speaking Part 1 questions about the topic '{topic}'.
        Return the response as a valid JSON array of strings only. Do not include any other text or markdown.
        Example for topic 'Hometown':
        [
            "Let's talk about your hometown. Where is it located?",
            "What do you like most about your hometown?",
            "Is there anything you dislike about it?",
            "How has your hometown changed over the years?"
        ]
        """
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        questions = json.loads(cleaned_response)
        
        print(f"Successfully generated questions for '{topic}': {questions}")
        return JSONResponse(content={"questions": questions})
    except Exception as e:
        print(f"ERROR generating IELTS questions with model '{model_name}': {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to generate IELTS questions."})

@app.post("/analyze-answer")
async def analyze_answer(question: str = Form(...), audio_file: UploadFile = File(...)):
    temp_in_path = ""
    wav_path = ""
    model_name = 'gemini-pro-latest'
    try:
        # 1. LƯU VÀ CHUYỂN ĐỔI AUDIO
        suffix = os.path.splitext(audio_file.filename)[1] if audio_file.filename else ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
            temp_in_path = tmp_in.name
            content = await audio_file.read()
            tmp_in.write(content)

        wav_path = temp_in_path.replace(suffix, ".wav")
        process = subprocess.run(
            ["ffmpeg", "-y", "-i", temp_in_path, "-ar", "16000", "-ac", "1", "-vn", wav_path],
            capture_output=True, text=True
        )
        if process.returncode != 0:
            print(f"FFMPEG Error: {process.stderr}")
            return JSONResponse(status_code=400, content={"error": "FFmpeg audio conversion failed."})

        # 2. SPEECH-TO-TEXT
        speech_config = speechsdk.SpeechConfig(subscription=os.getenv("SPEECH_KEY"), region=os.getenv("SPEECH_REGION"))
        audio_config = speechsdk.audio.AudioConfig(filename=wav_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        result = recognizer.recognize_once_async().get()
        
        user_transcription = result.text if result.reason == speechsdk.ResultReason.RecognizedSpeech else ""

        if not user_transcription.strip() or user_transcription.strip() == ".":
            return JSONResponse(content={
                "user_transcription": "I couldn't hear you clearly. Please try again.",
                "feedback": "No valid answer was detected.", "score": 0,
                "model_answer": "Please record your answer again.", "audio_url": None
            })

        # 3. GỌI GEMINI ĐỂ ĐÁNH GIÁ
        print(f"Attempting to use confirmed available model: '{model_name}'")
        model = genai.GenerativeModel(model_name)
        prompt = f"""
        As an expert IELTS examiner, evaluate a student's answer for a Speaking Part 1 question.
        Provide your response as a valid JSON object with three keys: "feedback", "score", and "model_answer".

        Question: "{question}"
        Student's Answer: "{user_transcription}"

        JSON Structure:
        - "feedback": (string) Provide constructive feedback on fluency, vocabulary, grammar, and coherence. Start with a positive comment.
        - "score": (number) Estimate an IELTS band score for this answer, from 5.0 to 9.0.
        - "model_answer": (string) Provide a concise, high-quality Band 9 sample answer for the question.
        """
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        ai_analysis = json.loads(cleaned_response)

        # 4. LƯU FILE AUDIO
        audio_dir = "static/audio"
        os.makedirs(audio_dir, exist_ok=True)
        student_audio_filename = os.path.basename(wav_path)
        final_audio_path = os.path.join(audio_dir, student_audio_filename)
        os.rename(wav_path, final_audio_path)
        
        return JSONResponse(content={
            "user_transcription": user_transcription,
            "feedback": ai_analysis.get("feedback"),
            "score": ai_analysis.get("score"),
            "model_answer": ai_analysis.get("model_answer"),
            "audio_url": f"/{final_audio_path}"
        })

    except Exception as e:
        print(f"ERROR in /analyze-answer with model '{model_name}': {e}")
        return JSONResponse(status_code=500, content={"error": "An internal server error occurred during analysis."})
    finally:
        if temp_in_path and os.path.exists(temp_in_path):
            os.remove(temp_in_path)
        if 'final_audio_path' not in locals() and wav_path and os.path.exists(wav_path):
            os.remove(wav_path)

# --- CHỨC NĂNG DỊCH FILE ---
@app.post("/translate-file")
async def translate_file(file: UploadFile = File(...)):
    contents = await file.read()
    original_text = ""

    try:
        content_type = file.content_type
        # --- CẬP NHẬT: SỬ DỤNG AZURE AI VISION CHO ẢNH ---
        if content_type in ["image/jpeg", "image/png", "image/webp"]:
            print("Processing image file using Azure AI Vision...")
            
            vision_key = os.getenv("VISION_KEY")
            vision_endpoint = os.getenv("VISION_ENDPOINT")

            if not vision_key or not vision_endpoint:
                raise Exception("Azure Vision Key or Endpoint is not configured in .env file.")

            computervision_client = ComputerVisionClient(vision_endpoint, CognitiveServicesCredentials(vision_key))
            image_stream = io.BytesIO(contents)
            
            read_response = computervision_client.read_in_stream(image_stream, raw=True)
            operation_location = read_response.headers["Operation-Location"]
            operation_id = operation_location.split("/")[-1]

            while True:
                read_result = computervision_client.get_read_result(operation_id)
                if read_result.status.lower() not in ['notstarted', 'running']:
                    break
                print("Waiting for Azure to process the image...")
                time.sleep(1)

            text_lines = []
            if read_result.status.lower() == 'succeeded':
                for text_result in read_result.analyze_result.read_results:
                    for line in text_result.lines:
                        text_lines.append(line.text)
            
            original_text = "\n".join(text_lines)
        # --- KẾT THÚC CẬP NHẬT ---
        elif file.filename.endswith(".txt"):
            print("Processing text file...")
            original_text = contents.decode("utf-8")
        elif file.filename.endswith(".docx"):
            print("Processing docx file...")
            doc = docx.Document(io.BytesIO(contents))
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            original_text = '\n'.join(full_text)
        else:
            return JSONResponse(status_code=400, content={"error": f"Unsupported file type: {content_type}."})

        if not original_text.strip():
            return JSONResponse(content={
                "original_text": "[No text found in the file]",
                "translated_text": "Không tìm thấy văn bản nào trong file để dịch."
            })

        # --- DỊCH VĂN BẢN VẪN DÙNG GEMINI ---
        print("Translating extracted text using Gemini...")
        text_model_name = 'gemini-pro-latest'
        model = genai.GenerativeModel(text_model_name)
        prompt = f"Translate the following text to Vietnamese:\n\n---\n\n{original_text}"
        response = model.generate_content(prompt)
        translated_text = response.text

        return JSONResponse(content={
            "original_text": original_text,
            "translated_text": translated_text
        })
    
    except google_exceptions.ResourceExhausted as e:
        print(f"QUOTA ERROR for Gemini translation: {e}")
        return JSONResponse(status_code=429, content={"error": "Bạn đã dùng hết lượt dịch miễn phí của Google. Vui lòng thử lại sau."})
    except Exception as e:
        print(f"ERROR in /translate-file: {e}")
        return JSONResponse(status_code=500, content={"error": "An internal server error occurred."})

# --- CHỨC NĂNG MỚI: CHUYỂN ÂM THANH THÀNH VĂN BẢN ---
@app.post("/transcribe-audio")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """
    Nhận file âm thanh, chuyển đổi thành văn bản bằng Azure Speech to Text.
    """
    temp_in_path = ""
    wav_path = ""
    try:
        # 1. LƯU VÀ CHUYỂN ĐỔI AUDIO SANG ĐỊNH DẠNG WAV
        suffix = os.path.splitext(audio_file.filename)[1] if audio_file.filename else ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
            temp_in_path = tmp_in.name
            content = await audio_file.read()
            tmp_in.write(content)

        wav_path = temp_in_path.replace(suffix, ".wav")
        process = subprocess.run(
            ["ffmpeg", "-y", "-i", temp_in_path, "-ar", "16000", "-ac", "1", "-vn", wav_path],
            capture_output=True, text=True
        )
        if process.returncode != 0:
            print(f"FFMPEG Error: {process.stderr}")
            return JSONResponse(status_code=400, content={"error": "FFmpeg audio conversion failed."})

        # 2. CHUYỂN GIỌNG NÓI THÀNH VĂN BẢN BẰNG AZURE
        speech_key = os.getenv("SPEECH_KEY")
        speech_region = os.getenv("SPEECH_REGION")
        if not speech_key or not speech_region:
            return JSONResponse(status_code=500, content={"error": "Azure Speech credentials not set."})

        speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
        audio_config = speechsdk.audio.AudioConfig(filename=wav_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        
        print("Transcribing audio with Azure...")
        result = recognizer.recognize_once_async().get()
        
        transcribed_text = ""
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            transcribed_text = result.text
        elif result.reason == speechsdk.ResultReason.NoMatch:
            transcribed_text = "[No speech could be recognized]"
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Speech recognition canceled: {cancellation_details.reason}")
            if cancellation_details.reason == speechsdk.CancellationReason.Error:
                print(f"Error details: {cancellation_details.error_details}")
            return JSONResponse(status_code=500, content={"error": "Speech recognition was canceled."})

        return JSONResponse(content={"transcription": transcribed_text})

    except Exception as e:
        print(f"ERROR in /transcribe-audio: {e}")
        return JSONResponse(status_code=500, content={"error": "An internal server error occurred during transcription."})
    finally:
        # Xóa các file tạm
        if temp_in_path and os.path.exists(temp_in_path):
            os.remove(temp_in_path)
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


# --- ENDPOINTS KHÁC ---

@app.post("/tts")
async def tts_endpoint(text: str = Form(...)):
    speech_key = os.getenv("SPEECH_KEY")
    speech_region = os.getenv("SPEECH_REGION")
    if not speech_key or not speech_region:
        return JSONResponse(status_code=500, content={"error": "Azure Speech credentials not set."})

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
    speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        file_path = tmp.name

    audio_output = speechsdk.audio.AudioConfig(filename=file_path)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output)
    result = synthesizer.speak_text_async(text).get()

    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        return JSONResponse(status_code=500, content={"error": str(result.cancellation_details)})

    return FileResponse(file_path, media_type="audio/mpeg", filename="tts.mp3", background=lambda: os.remove(file_path))

@app.websocket("/ws/stt")
async def websocket_stt(websocket: WebSocket):
    await websocket.accept()
    ref_text = websocket.query_params.get("ref_text", "Hello world.")
    webm_path, wav_path = "", ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_webm:
            webm_path = tmp_webm.name
            while True:
                message = await websocket.receive()
                if "text" in message and message.get("text") == "END": break
                elif "bytes" in message: tmp_webm.write(message["bytes"])

        wav_path = webm_path.replace(".webm", ".wav")
        process = subprocess.run(
            ["ffmpeg", "-y", "-i", webm_path, "-ar", "16000", "-ac", "1", "-vn", wav_path],
            capture_output=True, text=True
        )
        if process.returncode != 0:
            await websocket.send_json({"final": True, "error": "FFmpeg audio conversion failed."})
            return

        recognition_result = perform_recognition(wav_path, ref_text)
        await websocket.send_json({"final": True, **recognition_result})

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"WebSocket Error: {e}")
    finally:
        if websocket.client_state.name != "DISCONNECTED": await websocket.close()
        if webm_path and os.path.exists(webm_path): os.remove(webm_path)
        if wav_path and os.path.exists(wav_path): os.remove(wav_path)

# --- PHỤC VỤ FILE TĨNH ---
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_home():
    return FileResponse("static/index.html")

