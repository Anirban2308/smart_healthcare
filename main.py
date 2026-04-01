
"""Smart Healthcare App — Backend API
Phase 1: Prescription Summarization Endpoint"""



from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import re

app = FastAPI(
    title="Smart Healthcare API",
    description="Backend for Smart Healthcare Android App — Rural Communities",
    version="1.0.0-phase1"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



class PrescriptionRequest(BaseModel):
    prescription_text: str

class Medicine(BaseModel):
    name: str
    dose: str
    frequency: str
    duration: str
    instruction: str

class PrescriptionSummary(BaseModel):
    patient_summary: str
    medicines: List[Medicine]
    warnings: List[str]



FREQUENCY_MAP = {
    "OD":    "Once daily",
    "BD":    "Twice daily",
    "TDS":   "Three times daily",
    "QID":   "Four times daily",
    "SOS":   "When needed (if pain/fever)",
    "HS":    "At bedtime",
    "1-0-1": "Morning and night",
    "1-1-1": "Three times a day",
    "1-0-0": "Once daily in the morning",
    "0-0-1": "Once daily at night",
    "0-1-0": "Once daily at noon",
}

STANDARD_WARNINGS = [
    "Complete the full course even if symptoms improve.",
    "Do not take medicines on an empty stomach unless instructed.",
    "Consult your doctor immediately if rash, swelling, or difficulty breathing occurs.",
    "Store all medicines away from sunlight and out of reach of children.",
]



def parse_medicine_line(line: str) -> Medicine | None:
    line = line.strip()
    if not re.search(r'\b(Tab|Cap|Syp|Inj|Drop|Oint)\.?\b', line, re.IGNORECASE):
        return None

   
    name_match = re.search(
        r'(Tab|Cap|Syp|Inj|Drop|Oint)\.?\s+([\w\s\-]+?)(\d+\s*(?:mg|ml|mcg|g))?(?:\s+[\-—]|\s+\d|\s+x|\s+OD|\s+BD|\s+TDS|$)',
        line, re.IGNORECASE
    )
    name = "Unknown Medicine"
    dose = "As prescribed"
    if name_match:
        med_type = name_match.group(1).capitalize()
        med_name = name_match.group(2).strip().title()
        strength = name_match.group(3) or ""
        name = f"{med_type}. {med_name} {strength}".strip()
        dose = strength if strength else "1 unit"

 
    frequency = "As directed by doctor"
    for abbr, label in FREQUENCY_MAP.items():
        if re.search(rf'\b{re.escape(abbr)}\b', line, re.IGNORECASE):
            frequency = label
            break

   
    dur_match = re.search(r'x?\s*(\d+)\s*days?', line, re.IGNORECASE)
    duration = f"{dur_match.group(1)} days" if dur_match else "As prescribed"

    
    inst_match = re.search(r'\(([^)]+)\)', line)
    instruction = inst_match.group(1).capitalize() if inst_match else "As directed"

    return Medicine(
        name=name,
        dose=dose,
        frequency=frequency,
        duration=duration,
        instruction=instruction
    )



def build_patient_summary(medicines: List[Medicine]) -> str:
    if not medicines:
        return "Please follow your doctor's prescription instructions carefully."
    parts = [
        f"Take {m.name} {m.frequency.lower()} for {m.duration}. {m.instruction}."
        for m in medicines
    ]
    return " ".join(parts)



@app.get("/")
def root():
    return {
        "service": "Smart Healthcare API",
        "phase": "Phase 1 — Prescription Summarization",
        "status": "operational"
    }

@app.post("/summarize-prescription", response_model=PrescriptionSummary)
def summarize_prescription(request: PrescriptionRequest):
    """
    Accepts raw prescription text and returns a structured, simplified summary.

    Phase 1: Rule-based NLP with pattern matching.
    Phase 2: Will be upgraded to Claude API for contextual understanding,
             multilingual output (Hindi/English/Odia), and OCR pipeline.
    """
    text = request.prescription_text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Prescription text cannot be empty.")

    medicines = []
    for line in text.splitlines():
        med = parse_medicine_line(line)
        if med:
            medicines.append(med)

    patient_summary = build_patient_summary(medicines)


    warnings = list(STANDARD_WARNINGS)
    if any("amoxicillin" in m.name.lower() or "azithromycin" in m.name.lower() for m in medicines):
        warnings.insert(0, "Antibiotic detected: Do not stop the course early even if you feel better.")

    return PrescriptionSummary(
        patient_summary=patient_summary,
        medicines=medicines,
        warnings=warnings[:4]
    )



@app.get("/sample-response")
def sample_response():
    """Returns a hardcoded sample response for UI testing without backend."""
    return {
        "patient_summary": (
            "Take Paracetamol 500mg twice daily after food for 5 days. "
            "Take Amoxicillin 250mg three times daily with water for 7 days. "
            "Take Pantoprazole 40mg once daily in the morning before food for 7 days."
        ),
        "medicines": [
            {"name": "Tab. Paracetamol 500mg", "dose": "500mg", "frequency": "Twice daily", "duration": "5 days", "instruction": "After food"},
            {"name": "Cap. Amoxicillin 250mg", "dose": "250mg", "frequency": "Three times daily", "duration": "7 days", "instruction": "With water"},
            {"name": "Tab. Pantoprazole 40mg", "dose": "40mg", "frequency": "Once daily in the morning", "duration": "7 days", "instruction": "Before food"},
        ],
        "warnings": [
            "Antibiotic detected: Do not stop the course early even if you feel better.",
            "Complete the full course even if symptoms improve.",
            "Consult your doctor immediately if rash, swelling, or difficulty breathing occurs.",
        ]
    }
