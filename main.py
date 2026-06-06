import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from gtts import gTTS
import uuid
import cohere
from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile, File
from PIL import Image
import pytesseract
from datetime import datetime, timedelta
from typing import Optional


load_dotenv()

app = FastAPI(title="Smart Healthcare Android Application Backend")

# create audio folder automatically
os.makedirs("audio", exist_ok=True)

app.mount("/audio", StaticFiles(directory="audio"), name="audio")

co = cohere.ClientV2(os.getenv("COHERE_API_KEY"))
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class PrescriptionRequest(BaseModel):
    prescription_text: str = Field(..., min_length=5)
    language: str = Field(..., description="English, Hindi, or Odia")


@app.get("/")
def home():
    return {
        "message": "Smart Healthcare Backend is running",
        "ai_mode": "LLM-based prescription summarization"
    }


@app.post("/summarize-prescription")
def summarize_prescription(request: PrescriptionRequest):
    try:
        prompt = f"""
You are a healthcare assistant for rural patients.

Summarize the following prescription in simple {request.language}.
Do not diagnose. Do not invent medicines.
Return ONLY valid JSON with exactly these keys:
- summary: string, overall summary
- medicines: list of objects, each with keys: name, dosage, frequency, duration, instructions
- dosage_instructions: string, general dosage guidance
- safety_warnings: string, important warnings
- voice_text: string, simple text to be read aloud
- language: string

Prescription:
{request.prescription_text}
"""

        response = co.chat(
            model="command-r-plus-08-2024",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"}
        )

        result_text = response.message.content[0].text
        result_json = json.loads(result_text)

        voice_text = result_json["voice_text"]

        language_map = {
          "English": "en",
          "Hindi": "hi",
          "Odia": "hi"
        }

        tts_lang = language_map.get(request.language, "en")

        audio_filename = f"{uuid.uuid4()}.mp3"
        audio_path = f"audio/{audio_filename}"

        tts = gTTS(text=voice_text, lang=tts_lang)
        tts.save(audio_path)

        result_json["audio_file"] = audio_filename

        return {
            "status": "success",
            "processing_type": "LLM_PROCESS_AI_ENGINE",
            "data": result_json
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/upload-prescription-image")
async def upload_prescription_image(
         file: UploadFile = File(...)
):
    try:
        image = Image.open(file.file)

        extracted_text = pytesseract.image_to_string(image)

        return {
            "status": "success",
            "extracted_text": extracted_text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/upload-and-summarize")
async def upload_and_summarize(
    file: UploadFile = File(...),
    language: str = "English"
):
    try:
        image = Image.open(file.file)
        extracted_text = pytesseract.image_to_string(image).strip()

        if len(extracted_text) < 5:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from image. Please use a clearer image."
            )

        request = PrescriptionRequest(
            prescription_text=extracted_text,
            language=language
        )
        return summarize_prescription(request)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

class AlternativeMedicineRequest(BaseModel):
    medicine_name: str
    language: str = "English"


@app.post("/alternative-medicine")
def alternative_medicine(request: AlternativeMedicineRequest):
    try:
        prompt = f"""
You are a healthcare assistant for rural users.

Suggest generic or common alternative medicine information for:
{request.medicine_name}

Use simple {request.language}.
Do not give unsafe medical advice.
Tell the user to confirm with a doctor or pharmacist before replacing medicine.

Return ONLY valid JSON.

JSON format:
{{
  "medicine_name": "string",
  "generic_name": "string",
  "possible_alternatives": ["string"],
  "use_case": "string",
  "dosage_guidance": "single plain text string",
  "safety_warning": "string",
  "language": "string"
}}

IMPORTANT:
- dosage_guidance MUST be a single string.
- Do NOT return dosage_guidance as an object or array.
- Do NOT include markdown.
- Do NOT include explanation outside JSON.

For dosage_guidance include:
typical dosage, frequency, and duration in one sentence.
"""

        response = co.chat(
            model="command-r-plus-08-2024",
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        result_text = response.message.content[0].text
        result_json = json.loads(result_text)

        return {
            "status": "success",
            "processing_type": "LLM_ALTERNATIVE_MEDICINE_ENGINE",
            "data": result_json
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
class DiseaseInfoRequest(BaseModel):
    disease_name: str
    disease_category: str
    user_type: str = "registered"
    language: str = "English"


@app.post("/disease-info")
def disease_info(request: DiseaseInfoRequest):
    try:
        word_limit = 300 if request.user_type == "registered" else 100

        prompt = f"""
You are a rural healthcare education assistant.

The user selected category: {request.disease_category}
The user is asking about: {request.disease_name}

Valid categories and their diseases:
- Viral Diseases: Dengue, COVID-19, Influenza, Chickenpox, Hepatitis B, and other viral infections
- Heart Diseases: Hypertension, Heart Attack, Coronary Artery Disease, Arrhythmia, and other heart conditions
- Brain Disorders: Migraine, Epilepsy, Parkinson's Disease, Stroke, and other neurological conditions
- Kidney-related Diseases: Kidney Stone, Chronic Kidney Disease, Urinary Tract Infection, Kidney Failure, and other kidney conditions

Rules:
1. Be flexible with spelling and case:
   - "dengue" = "Dengue"
   - "covid" = "COVID-19"
   - "heart attack" = "Heart Attack"
   - "kidney stone" = "Kidney Stone"
   - "migraine" = "Migraine"
   - "parkinson" = "Parkinson's Disease"
2. If the disease belongs to the selected category OR is clearly related to it, proceed with explanation
3. If the disease clearly does NOT belong to the selected category, return error

Examples of correct matches:
- "dengue" under "Viral Diseases" → VALID
- "heart attack" under "Heart Diseases" → VALID
- "kidney stone" under "Kidney-related Diseases" → VALID
- "migraine" under "Brain Disorders" → VALID
- "diabetes" under "Viral Diseases" → INVALID

If INVALID return ONLY this JSON:
{{
    "error": "true",
    "message": "In {request.language}: explain that {request.disease_name} does not belong to {request.disease_category} and suggest the correct category"
}}

If VALID explain the disease in simple {request.language}.
Word limit: {word_limit} words.

Include:
- basic explanation
- common symptoms
- prevention
- diet advice
- when to consult doctor

Return ONLY valid JSON with these keys:
disease_name, explanation, symptoms, prevention, diet_advice, doctor_advice, user_type, language, error.
Set error to "false" if valid disease.
User type: {request.user_type}
"""

        response = co.chat(
            model="command-r-plus-08-2024",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )

        result_json = json.loads(response.message.content[0].text)

        if result_json.get("error") == "true":
            raise HTTPException(
                status_code=400,
                detail=result_json.get("message", "Disease does not belong to selected category")
            )

        return {
            "status": "success",
            "processing_type": "LLM_DISEASE_INFO_ENGINE",
            "data": result_json
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
class AppointmentRequest(BaseModel):
    patient_name: str
    symptoms: str
    disease_category: str
    appointment_type: str  # quick or scheduled
    preferred_datetime: Optional[str] = None
    language: str = "English"

appointments = []

@app.post("/book-appointment")
def book_appointment(request: AppointmentRequest):
    appointment_id = str(uuid.uuid4())

    if request.appointment_type.lower() == "quick":
        scheduled_time = datetime.now() + timedelta(minutes=30)
        message = "Quick consultation request created. Doctor will respond within 30 minutes."

    elif request.appointment_type.lower() == "scheduled":
        if not request.preferred_datetime:
            raise HTTPException(
                status_code=400,
                detail="preferred_datetime is required for scheduled appointment"
            )
        scheduled_time = request.preferred_datetime
        message = "Scheduled appointment request created. Waiting for doctor confirmation."

    else:
        raise HTTPException(
            status_code=400,
            detail="appointment_type must be quick or scheduled"
        )

    appointment = {
        "appointment_id": appointment_id,
        "patient_name": request.patient_name,
        "symptoms": request.symptoms,
        "disease_category": request.disease_category,
        "appointment_type": request.appointment_type,
        "scheduled_time": str(scheduled_time),
        "status": "Pending",
        "doctor_response": "Waiting for doctor response",
        "message": message
    }

    appointments.append(appointment)

    return {
        "status": "success",
        "data": appointment
    }

@app.get("/doctor/appointments")
def doctor_appointments():
    return {
        "status": "success",
        "appointments": appointments
    }

@app.post("/doctor/respond-appointment/{appointment_id}")
def respond_appointment(appointment_id: str, response: str):
    for appointment in appointments:
        if appointment["appointment_id"] == appointment_id:
            if response.lower() == "accept":
                appointment["status"] = "Accepted"
                appointment["doctor_response"] = "Doctor accepted the appointment"
            elif response.lower() == "reject":
                appointment["status"] = "Rejected"
                appointment["doctor_response"] = "Doctor rejected the appointment"
            else:
                raise HTTPException(
                    status_code=400,
                    detail="response must be accept or reject"
                )

            return {
                "status": "success",
                "data": appointment
            }

    raise HTTPException(status_code=404, detail="Appointment not found")

class PersonalizedRecommendationRequest(BaseModel):
    medical_history: str
    language: str = "English"

@app.post("/personalized-recommendations")
def personalized_recommendations(
    request: PersonalizedRecommendationRequest
):
    try:

        prompt = f"""
You are a healthcare assistant for rural patients.

Based on the patient's medical history below,
generate personalized health recommendations.

Medical History:
{request.medical_history}

Provide:
1. Diet recommendations
2. Lifestyle recommendations
3. Preventive care suggestions
4. Follow-up advice

Use simple {request.language}.

Return ONLY valid JSON:

{{
  "diet_recommendations": "string",
  "lifestyle_recommendations": "string",
  "preventive_care": "string",
  "follow_up_advice": "string",
  "language": "string"
}}
"""

        response = co.chat(
            model="command-r-plus-08-2024",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"}
        )

        result_json = json.loads(
            response.message.content[0].text
        )

        return {
            "status": "success",
            "processing_type": "LLM_PERSONALIZED_RECOMMENDATION_ENGINE",
            "data": result_json
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )