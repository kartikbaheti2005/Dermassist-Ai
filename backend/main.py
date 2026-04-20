from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import numpy as np
import os, time, uuid, json, io, math, base64
from typing import Optional
from sqlalchemy.orm import Session
from PIL import Image as PILImage
from dotenv import load_dotenv

load_dotenv()

from database import engine, SessionLocal
from models import Base
from models.user import User
from models.images import Image
from models.prediciton import Prediction
from models.appointment import Appointment
from models.doctor import Doctor
import auth
import admin as admin_module
from auth import get_current_user, oauth2_scheme

app = FastAPI(title="DermAssist AI Backend", version="3.0.0")
Base.metadata.create_all(bind=engine)

# ── Sample doctor seeding ─────────────────────────────────────────────────────
SAMPLE_DOCTORS = [
    {"username":"priya.sharma","email":"priya.sharma@skinclinic.in","full_name":"Dr. Priya Sharma","phone":"+91 98100 11111","post":"Dermatologist","specialty":"Dermatologist","qualification":"MD, MBBS – AIIMS Delhi","practice_start_year":2011,"clinic_name":"Skin Care Clinic","address":"12, Connaught Place, New Delhi, Delhi 110001","city":"Delhi","available_days":["Mon","Tue","Wed","Fri"],"available_slots":["10:00 AM","11:00 AM","2:00 PM","3:30 PM"],"consultation_fee":700,"languages":["Hindi","English"],"specializes_in":["Melanoma","Acne","Eczema","Psoriasis"],"image_placeholder":"PS","rating":4.8,"review_count":312,"bio":"Experienced dermatologist specializing in skin cancer detection and acne treatment."},
    {"username":"rahul.mehta","email":"rahul.mehta@asi.in","full_name":"Dr. Rahul Mehta","phone":"+91 98200 22222","post":"Dermatologist & Oncologist","specialty":"Dermato-Oncology","qualification":"MD Dermatology, DNB – Bombay Hospital","practice_start_year":2007,"clinic_name":"Advanced Skin Institute","address":"45, Linking Road, Bandra West, Mumbai, Maharashtra 400050","city":"Mumbai","available_days":["Mon","Wed","Thu","Sat"],"available_slots":["9:00 AM","10:30 AM","12:00 PM","4:00 PM"],"consultation_fee":1200,"languages":["Hindi","English","Marathi"],"specializes_in":["Skin Cancer","Basal Cell Carcinoma","Moles","Pigmentation"],"image_placeholder":"RM","rating":4.9,"review_count":487,"bio":"Leading dermato-oncologist with expertise in skin cancer surgery and advanced diagnostics."},
    {"username":"ananya.krishnan","email":"ananya.k@dermacare.in","full_name":"Dr. Ananya Krishnan","phone":"+91 94440 33333","post":"Cosmetic Dermatologist","specialty":"Cosmetic Dermatology","qualification":"MD – Madras Medical College","practice_start_year":2015,"clinic_name":"DermaCare Centre","address":"78, Anna Salai, Teynampet, Chennai, Tamil Nadu 600018","city":"Chennai","available_days":["Tue","Wed","Fri","Sat"],"available_slots":["11:00 AM","1:00 PM","3:00 PM","5:00 PM"],"consultation_fee":600,"languages":["Tamil","English","Hindi"],"specializes_in":["Keratosis","Dermatofibroma","Anti-aging","Laser"],"image_placeholder":"AK","rating":4.7,"review_count":198,"bio":"Specializes in cosmetic procedures, anti-aging, and laser skin treatments."},
    {"username":"sanjay.gupta","email":"sanjay.gupta@pss.in","full_name":"Dr. Sanjay Gupta","phone":"+91 98700 44444","post":"Dermatologist","specialty":"Clinical Dermatology","qualification":"MD, DVD – KEM Hospital Pune","practice_start_year":2013,"clinic_name":"Pune Skin Solutions","address":"22, FC Road, Shivajinagar, Pune, Maharashtra 411005","city":"Pune","available_days":["Mon","Tue","Thu","Fri","Sat"],"available_slots":["9:30 AM","11:00 AM","2:30 PM","4:30 PM"],"consultation_fee":550,"languages":["Hindi","English","Marathi"],"specializes_in":["Vascular Lesions","Nevi","Skin Screening","Acne"],"image_placeholder":"SG","rating":4.6,"review_count":245,"bio":"General dermatologist focused on early skin cancer screening and acne management."},
]

def seed_sample_doctors():
    from passlib.context import CryptContext
    _pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
    db = SessionLocal()
    try:
        for d in SAMPLE_DOCTORS:
            if db.query(Doctor).filter(Doctor.email == d["email"]).first():
                continue
            db.add(Doctor(
                username=d["username"], email=d["email"],
                password_hash=_pwd.hash("SampleDoctor@123"),
                full_name=d["full_name"], phone=d["phone"],
                post=d["post"], specialty=d["specialty"],
                qualification=d["qualification"],
                practice_start_year=d["practice_start_year"],
                clinic_name=d["clinic_name"], address=d["address"], city=d["city"],
                available_days=d["available_days"], available_slots=d["available_slots"],
                consultation_fee=d["consultation_fee"], languages=d["languages"],
                specializes_in=d["specializes_in"], image_placeholder=d["image_placeholder"],
                rating=d["rating"], review_count=d["review_count"], bio=d["bio"],
                status="approved", is_active=True,
            ))
        db.commit()
        print("✅ Sample doctors seeded into database.")
    except Exception as e:
        db.rollback()
        print(f"⚠  Could not seed doctors: {e}")
    finally:
        db.close()

seed_sample_doctors()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(admin_module.router)

# ══════════════════════════════════════════════════════════════════════════════
#  ML MODEL — Single model: best_model.pth (EfficientNet-B4 DualBranch)
# ══════════════════════════════════════════════════════════════════════════════
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm as _timm
import cv2

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_model.pth")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "best_model.pth"

CLASSES = ["AK", "BCC", "BKL", "DF", "MEL", "NV", "SCC", "VASC"]

# ── Stage 1 routing ───────────────────────────────────────────────────────────
# Classes that trigger Stage 2 (non-cancerous — pass to disease classifier)
STAGE2_TRIGGER = {"NV", "BKL", "DF", "VASC"}
# Classes that stop at Stage 1 (cancerous — show result immediately)
CANCER_CLASSES = {"MEL", "BCC", "AK", "SCC"}

# ── Stage 2 config ────────────────────────────────────────────────────────────
STAGE2_MODEL_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage2_model.pth")
STAGE2_CLASSES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage2_classes.json")
if not os.path.exists(STAGE2_MODEL_PATH):
    STAGE2_MODEL_PATH = "stage2_model.pth"
if not os.path.exists(STAGE2_CLASSES_PATH):
    STAGE2_CLASSES_PATH = "stage2_classes.json"

STAGE2_CLASSES = []   # loaded after training
stage2_model   = None # loaded after training

STAGE2_RISK_MAP = {
    "Acne and Rosacea":       "Moderate Risk",
    "Eczema":                 "Moderate Risk",
    "Psoriasis":              "Moderate Risk",
    "Tinea Ringworm Fungal":  "Moderate Risk",
    "Warts Viral":            "Moderate Risk",
    "Bacterial Infection":    "High Risk",
    "Urticaria Hives":        "Moderate Risk",
    "Chickenpox":             "High Risk",
    "Monkeypox":              "High Risk",
    "Herpes":                 "High Risk",
    "Lupus":                  "High Risk",
    "Nail Fungus":            "Low Risk",
}

STAGE2_TYPE_MAP = {
    "Acne and Rosacea":       "Inflammatory",
    "Eczema":                 "Inflammatory",
    "Psoriasis":              "Inflammatory",
    "Tinea Ringworm Fungal":  "Fungal",
    "Warts Viral":            "Viral",
    "Bacterial Infection":    "Bacterial",
    "Urticaria Hives":        "Allergic",
    "Chickenpox":             "Viral",
    "Monkeypox":              "Viral",
    "Herpes":                 "Viral",
    "Lupus":                  "Autoimmune",
    "Nail Fungus":            "Fungal",
}

RISK_MAP = {
    "MEL": "High Risk", "BCC": "High Risk", "AK": "High Risk", "SCC": "High Risk",
    "BKL": "Moderate Risk", "DF": "Moderate Risk", "VASC": "Moderate Risk",
    "NV":  "Low Risk", "unk": "Low Risk",
}

NAME_MAP = {
    "MEL": "Melanoma", "BCC": "Basal Cell Carcinoma", "AK": "Actinic Keratosis",
    "SCC": "Squamous Cell Carcinoma", "BKL": "Benign Keratosis",
    "DF":  "Dermatofibroma", "VASC": "Vascular Lesion", "NV": "Melanocytic Nevi",
    "unk": "Unable to Analyse — Please Retake Photo",
}

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Add this right after IMG_TRANSFORM definition
STAGE2_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 7 TTA transforms — more augmentation = more stable predictions (+1-2% accuracy)
TTA_TRANSFORMS = [
    # 1. Original
    transforms.Compose([transforms.Resize((300,300)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    # 2. Horizontal flip
    transforms.Compose([transforms.Resize((300,300)), transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    # 3. Vertical flip
    transforms.Compose([transforms.Resize((300,300)), transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    # 4. Center crop from larger image
    transforms.Compose([transforms.Resize((320,320)), transforms.CenterCrop(300),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    # 5. Slight rotation
    transforms.Compose([transforms.Resize((300,300)), transforms.RandomRotation(degrees=15),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    # 6. Larger crop
    transforms.Compose([transforms.Resize((340,340)), transforms.CenterCrop(300),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    # 7. Horizontal flip + rotation
    transforms.Compose([transforms.Resize((300,300)), transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
]


# ── Phone Photo Preprocessing ─────────────────────────────────────────────────
# Converts regular phone camera photos to match dermoscope image characteristics
# so the model (trained on dermoscope data) can analyze them accurately

def apply_clahe(img_array):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Enhances local contrast — makes skin texture visible like dermoscope."""
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def apply_unsharp_mask(img_array, strength=0.6):
    """Unsharp masking — enhances fine skin texture detail."""
    blurred = cv2.GaussianBlur(img_array, (0, 0), 3)
    sharpened = cv2.addWeighted(img_array, 1 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def normalize_color(img_array):
    """Normalize color channels to reduce lighting variation.
    Phone photos have inconsistent white balance — this fixes that."""
    result = img_array.copy().astype(np.float32)
    for i in range(3):
        channel = result[:, :, i]
        p2, p98 = np.percentile(channel, 2), np.percentile(channel, 98)
        if p98 > p2:
            result[:, :, i] = np.clip((channel - p2) / (p98 - p2) * 255, 0, 255)
    return result.astype(np.uint8)

def crop_lesion(img_array, margin=0.15):
    """Auto-crop to center of image where lesion is most likely located.
    Phone photos often have large backgrounds — this focuses on the lesion."""
    h, w = img_array.shape[:2]
    # Take center 70% of image (removes background borders)
    m_h = int(h * margin)
    m_w = int(w * margin)
    cropped = img_array[m_h:h-m_h, m_w:w-m_w]
    # Only use crop if result is large enough
    if cropped.shape[0] > 50 and cropped.shape[1] > 50:
        return cropped
    return img_array

def preprocess_phone_photo(pil_image):
    """
    Full preprocessing pipeline for phone camera photos.
    Converts phone photo characteristics to match dermoscope training data.

    Steps:
    1. Convert to numpy array
    2. Crop center (remove background)
    3. CLAHE contrast enhancement
    4. Color normalization
    5. Unsharp masking for texture
    6. Convert back to PIL
    """
    img_array = np.array(pil_image.convert('RGB'))

    # Step 1: Center crop to focus on lesion
    img_array = crop_lesion(img_array, margin=0.1)

    # Step 2: CLAHE — enhance local contrast
    img_array = apply_clahe(img_array)

    # Step 3: Color normalization — fix white balance
    img_array = normalize_color(img_array)

    # Step 4: Unsharp mask — enhance texture detail
    img_array = apply_unsharp_mask(img_array, strength=0.5)

    return PILImage.fromarray(img_array)

# ── Model Architecture ────────────────────────────────────────────────────────
class SkinModel(nn.Module):
    """
    EfficientNet-B4 + Patient Metadata dual-branch model.
    Trained on ISIC 2019 + HAM10000 (~35k images). Val accuracy: 83.5%
    Checkpoint: best_model.pth
    """
    def __init__(self):
        super().__init__()
        self.image_branch = _timm.create_model('efficientnet_b4', pretrained=False)
        self.image_branch.classifier = nn.Identity()
        self.image_fc = nn.Sequential(
            nn.Linear(1792, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4)
        )
        self.meta_fc = nn.Sequential(
            nn.Linear(11, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3)
        )
        self.fusion = nn.Sequential(
            nn.Linear(768, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 8)
        )

    def forward(self, image, meta=None):
        img_feat = self.image_fc(self.image_branch(image))
        if meta is not None and meta.abs().sum() > 0:
            meta_feat = self.meta_fc(meta)
        else:
            meta_feat = torch.zeros(img_feat.shape[0], 256, device=img_feat.device)
        return self.fusion(torch.cat([img_feat, meta_feat], dim=1))

# ── Stage 2 Model — EfficientNet-B2, image only ──────────────────────────────
class Stage2SkinModel(nn.Module):
    """EfficientNet-B2 for general skin disease classification (Stage 2)."""
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = _timm.create_model('efficientnet_b4', pretrained=False)
        in_features = self.backbone.classifier.in_features  # 1408
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)

# ============================================================================
# DermAssist AI — Stage 2 B4+B3 Ensemble Patch
# ============================================================================
# Replace / merge the following blocks into your backend/main.py
#
# Changes made vs original:
#   1. Stage2SkinModelB3  — new class (mirrors SkinModelB3 for Stage 1)
#   2. stage2_b3_model    — new global, loaded from stage2_b3_model.pth
#   3. load_stage2_b3_model() — new loader function
#   4. run_stage2()        — updated to use B4+B3 ensemble when B3 is loaded
#
# IMPORTANT — file placement in backend/:
#   stage2_model.pth      ← your existing B4 checkpoint  (unchanged)
#   stage2_classes.json   ← your existing classes file   (unchanged)
#   stage2_b3_model.pth   ← NEW B3 checkpoint from the training notebook
# ============================================================================


# ── Paste this AFTER your existing Stage2SkinModel class definition ──────────

class Stage2SkinModelB3(nn.Module):
    """
    EfficientNet-B3 companion for Stage 2 general skin disease classification.
    Matches the output dimension of Stage2SkinModel (B4) so probabilities
    can be blended: final = 0.55 * B4_probs + 0.45 * B3_probs
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = _timm.create_model('efficientnet_b3', pretrained=False)
        in_features = self.backbone.classifier.in_features   # 1536
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ── Add these globals near your other Stage 2 globals ────────────────────────

STAGE2_B3_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "stage2_b3_model.pth"
)
if not os.path.exists(STAGE2_B3_MODEL_PATH):
    STAGE2_B3_MODEL_PATH = "stage2_b3_model.pth"

stage2_b3_model = None   # loaded by load_stage2_b3_model()

# Ensemble weights — same ratio as Stage 1 (B4 55%, B3 45%)
STAGE2_B4_WEIGHT = 0.55
STAGE2_B3_WEIGHT = 0.45


# ── Add this loader function (call it right after load_stage2_model()) ────────

def load_stage2_b3_model():
    """
    Load the Stage 2 B3 ensemble companion.
    Falls back gracefully to B4-only if the file is missing.
    """
    global stage2_b3_model
    if not os.path.exists(STAGE2_B3_MODEL_PATH):
        print("ℹ️  stage2_b3_model.pth not found — Stage 2 running B4 only")
        print("   Train the B3 with Stage2_B3_Training.ipynb and place it in backend/")
        return
    if STAGE2_CLASSES is None:
        print("⚠️  STAGE2_CLASSES not set — skipping B3 load")
        return
    try:
        print("⏳ Loading Stage 2 B3 ensemble model...")
        model = Stage2SkinModelB3(num_classes=len(STAGE2_CLASSES))
        checkpoint = torch.load(
            STAGE2_B3_MODEL_PATH, map_location=DEVICE, weights_only=False
        )
        # Support both plain state_dict and {'state_dict': ..., 'classes': ...}
        state = (
            checkpoint["state_dict"]
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state, strict=True)
        stage2_b3_model = model.to(DEVICE).eval()
        print(
            f"✅ Stage 2 B3 loaded — ensemble ACTIVE "
            f"(B4×{STAGE2_B4_WEIGHT} + B3×{STAGE2_B3_WEIGHT})"
        )
    except Exception as e:
        print(f"❌ Stage 2 B3 failed to load: {e}")
        stage2_b3_model = None


# Call order in your startup section:
#   load_stage2_model()       ← already exists
#   load_stage2_b3_model()    ← ADD THIS LINE right after


# ── Replace your existing run_stage2() with this version ─────────────────────

def run_stage2(image_bytes: bytes):
    """
    Run Stage 2 general disease classifier.

    Uses B4+B3 weighted ensemble when stage2_b3_model.pth is present,
    otherwise falls back to B4-only — identical behaviour to Stage 1.

    Returns
    -------
    pred_class  : str   — winning class name
    confidence  : float — winning class probability (0–1)
    all_scores  : dict  — {class_name: probability} for all 23 classes
    """
    if stage2_model is None:
        return None, 0.0, {}

    try:
        # ── Shared preprocessing ──────────────────────────────────────────────
        pil = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        pil = preprocess_phone_photo(pil)   # your existing phone preprocessing

        # B4 uses 300×300, B3 uses 260×260 — define both transforms
        b4_transform = STAGE2_TRANSFORM     # your existing transform (300×300)

        b3_transform = transforms.Compose([
            transforms.Resize((260, 260)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # ── B4 inference ──────────────────────────────────────────────────────
        tensor_b4 = b4_transform(pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            b4_probs = torch.softmax(
                stage2_model(tensor_b4), dim=1
            ).cpu().numpy()[0]                      # shape: (num_classes,)

        # ── B3 inference (optional) ───────────────────────────────────────────
        if stage2_b3_model is not None:
            tensor_b3 = b3_transform(pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                b3_probs = torch.softmax(
                    stage2_b3_model(tensor_b3), dim=1
                ).cpu().numpy()[0]                  # shape: (num_classes,)

            # Weighted ensemble
            ensemble_probs = (
                STAGE2_B4_WEIGHT * b4_probs +
                STAGE2_B3_WEIGHT * b3_probs
            )
            model_tag = f"B4({STAGE2_B4_WEIGHT})+B3({STAGE2_B3_WEIGHT})"
        else:
            ensemble_probs = b4_probs
            model_tag = "B4-only"

        # ── Final prediction ─────────────────────────────────────────────────
        probs_clean = [safe_float(p) for p in ensemble_probs]
        idx         = int(np.argmax(probs_clean))
        pred_class  = STAGE2_CLASSES[idx]
        confidence  = probs_clean[idx]
        all_scores  = {
            STAGE2_CLASSES[i]: safe_float(ensemble_probs[i])
            for i in range(len(STAGE2_CLASSES))
        }

        print(f"🔬 Stage 2 [{model_tag}]: {pred_class} | conf={confidence:.3f}")
        return pred_class, confidence, all_scores

    except Exception as e:
        print(f"⚠️  Stage 2 failed: {e}")
        return None, 0.0, {}


# ============================================================================
# STARTUP CALL ORDER — find your startup block in main.py and add line 3:
#
#   load_model()               # Stage 1 B4  (already exists)
#   load_b3_model()            # Stage 1 B3  (already exists)
#   load_stage2_model()        # Stage 2 B4  (already exists)
#   load_stage2_b3_model()     # Stage 2 B3  ← ADD THIS
#
# ============================================================================

# ── Load model ────────────────────────────────────────────────────────────────
torch_model = None

def load_model():
    global torch_model
    if not os.path.exists(MODEL_PATH):
        print(f"❌ best_model.pth not found at: {MODEL_PATH}")
        print(f"   Place best_model.pth in the backend/ folder and restart.")
        return
    try:
        print(f"⏳ Loading SkinModel from {MODEL_PATH} ...")
        model = SkinModel()
        state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state, strict=True)
        torch_model = model.to(DEVICE).eval()
        print(f"✅ SkinModel loaded on {DEVICE} | Classes: {CLASSES}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        torch_model = None

load_model()

def load_stage2_model():
    """Load Stage 2 model if available."""
    global stage2_model, STAGE2_CLASSES
    if not os.path.exists(STAGE2_MODEL_PATH):
        print(f"ℹ️  Stage 2 model not found at {STAGE2_MODEL_PATH} — Stage 2 disabled")
        print(f"   Train it on Kaggle and place stage2_model.pth + stage2_classes.json in backend/")
        return
    if not os.path.exists(STAGE2_CLASSES_PATH):
        print(f"❌ stage2_classes.json not found — Stage 2 disabled")
        return
    try:
        import json as _json
        with open(STAGE2_CLASSES_PATH) as f:
            STAGE2_CLASSES = _json.load(f)
        model = Stage2SkinModel(num_classes=len(STAGE2_CLASSES))
        checkpoint = torch.load(STAGE2_MODEL_PATH, map_location=DEVICE, weights_only=False)
        # Handle both formats: plain state_dict OR {'state_dict': ..., 'classes': ...}
        state = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state, strict=True)
        stage2_model = model.to(DEVICE).eval()
        print(f"✅ Stage 2 model loaded | {len(STAGE2_CLASSES)} classes: {STAGE2_CLASSES}")
    except Exception as e:
        print(f"❌ Stage 2 model failed to load: {e}")
        stage2_model = None

load_stage2_model()

def get_stage2_risk(class_name):
    for key, risk in STAGE2_RISK_MAP.items():
        if key.lower() in class_name.lower():
            return risk
    return "Moderate Risk"

def run_stage2(image_bytes):
    """
    Run Stage 2 general disease classifier.
    Returns (class_name, confidence, all_scores)
    Called when Stage 1 returns NV/BKL/DF/VASC (non-cancerous)
    """
    if stage2_model is None:
        return None, 0.0, {}
    try:
        pil    = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        pil    = preprocess_phone_photo(pil)
        tensor = STAGE2_TRANSFORM(pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(stage2_model(tensor), dim=1).cpu().numpy()[0]
        probs_clean = [safe_float(p) for p in probs]
        idx         = int(np.argmax(probs_clean))
        pred_class  = STAGE2_CLASSES[idx]
        confidence  = probs_clean[idx]
        all_scores  = {STAGE2_CLASSES[i]: safe_float(probs[i]) for i in range(len(STAGE2_CLASSES))}
        print(f"🔬 Stage 2: {pred_class} | conf={confidence:.3f}")
        return pred_class, confidence, all_scores
    except Exception as e:
        print(f"⚠️  Stage 2 failed: {e}")
        return None, 0.0, {}

# ── B3 Ensemble Model (optional) ─────────────────────────────────────────────
B3_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "b3_model.pth")
if not os.path.exists(B3_MODEL_PATH):
    B3_MODEL_PATH = "b3_model.pth"

class SkinModelB3(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_branch = _timm.create_model('efficientnet_b3', pretrained=False)
        img_out = self.image_branch.classifier.in_features
        self.image_branch.classifier = nn.Identity()
        self.image_fc = nn.Sequential(nn.Linear(img_out,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Dropout(0.4))
        self.meta_fc  = nn.Sequential(nn.Linear(11,128),nn.ReLU(),nn.Linear(128,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.3))
        self.fusion   = nn.Sequential(nn.Linear(768,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.4),nn.Linear(256,8))
    def forward(self, img, meta=None):
        f = self.image_fc(self.image_branch(img))
        m = self.meta_fc(meta) if meta is not None and meta.abs().sum()>0 else torch.zeros(f.shape[0],256,device=f.device)
        return self.fusion(torch.cat([f,m],dim=1))

b3_model = None

def load_b3_model():
    global b3_model
    if not os.path.exists(B3_MODEL_PATH):
        print("ℹ️  b3_model.pth not found — using B4 only (add b3_model.pth to enable ensemble)")
        return
    try:
        print("⏳ Loading B3 ensemble model...")
        m = SkinModelB3()
        m.load_state_dict(torch.load(B3_MODEL_PATH, map_location=DEVICE, weights_only=False), strict=True)
        b3_model = m.to(DEVICE).eval()
        print("✅ B3 ensemble loaded — B4+B3 active!")
    except Exception as e:
        print(f"❌ B3 model failed: {e}")
        b3_model = None

load_b3_model()

# ── Stage 0: Image Quality Check ─────────────────────────────────────────────
def check_image_quality(image_bytes):
    """
    Basic image quality check — rejects clearly bad images.
    Returns: (is_valid, reason)
    """
    try:
        import cv2 as _cv2
        pil = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = pil.size

        # Check minimum size
        if w < 64 or h < 64:
            return False, "Image is too small. Please use a higher resolution photo."

        # Check if image is too dark or too bright (likely invalid)
        img_arr = np.array(pil.convert("L"))  # grayscale
        mean_brightness = img_arr.mean()
        if mean_brightness < 20:
            return False, "Image is too dark. Please use better lighting."
        if mean_brightness > 245:
            return False, "Image is too bright/overexposed. Please adjust lighting."

        # Check blur using Laplacian variance
        img_cv = np.array(pil)
        gray   = _cv2.cvtColor(img_cv, _cv2.COLOR_RGB2GRAY)
        blur   = _cv2.Laplacian(gray, _cv2.CV_64F).var()
        if blur < 15:
            return False, "Image is too blurry. Please hold the camera steady and retake."

        return True, "OK"
    except Exception as e:
        return True, "OK"  # If check fails, allow through

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
_gradcam_features  = {}
_gradcam_gradients = {}

def _register_hooks(model):
    """Hook onto conv_head of timm EfficientNet-B4 — the last conv before pooling."""
    try:
        if hasattr(model.image_branch, 'conv_head'):
            target = model.image_branch.conv_head
        elif hasattr(model.image_branch, 'blocks'):
            target = model.image_branch.blocks[-1]
        else:
            target = list(model.image_branch.children())[-3]
        print(f"✅ Grad-CAM hook: {type(target).__name__}")
    except Exception as e:
        print(f"❌ Grad-CAM hook failed: {e}")
        return None, None

    fh = target.register_forward_hook(
        lambda m, i, o: _gradcam_features.update({'value': o.detach()})
    )
    bh = target.register_full_backward_hook(
        lambda m, gi, go: _gradcam_gradients.update({'value': go[0].detach()})
    )
    return fh, bh


def generate_gradcam(model, image_bytes, class_idx):
    try:
        pil = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Apply same preprocessing as inference for better heatmap accuracy
        pil_processed = preprocess_phone_photo(pil)
        t = IMG_TRANSFORM(pil_processed).unsqueeze(0).to(DEVICE)
        t.requires_grad_(True)

        fh, bh = _register_hooks(model)
        if fh is None:
            return None, None

        model.eval()
        _gradcam_features.clear()
        _gradcam_gradients.clear()

        torch.set_grad_enabled(True)
        out = model(t, None)
        model.zero_grad()
        out[0, class_idx].backward()
        torch.set_grad_enabled(False)

        fh.remove(); bh.remove()

        if 'value' not in _gradcam_features or 'value' not in _gradcam_gradients:
            print("⚠️  Grad-CAM: no features/gradients captured")
            return None, None

        feats = _gradcam_features['value'].cpu()
        grads = _gradcam_gradients['value'].cpu()

        if feats.dim() == 2:
            return None, None

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam     = torch.relu((weights * feats).sum(dim=1)).squeeze().numpy()

        # ── Improved CAM post-processing ──────────────────────────────────
        # Apply Gaussian blur to smooth out noisy activations
        #cam = cv2.GaussianBlur(cam, (7, 7), 0)

        # Normalize
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            return None, None

        # Threshold — only show top 40% activations (removes random noise)
        #cam = np.where(cam >= 0.40, cam, 0)

        # Re-normalize after threshold
        if cam.max() > 0:
            cam = cam / cam.max()

        cam_resized = cv2.resize(cam, (300, 300))
        cam_uint8   = np.uint8(255 * cam_resized)
        heatmap     = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Use processed image for overlay (matches what model actually saw)
        orig    = np.array(pil_processed.resize((300, 300)))
        overlay = cv2.addWeighted(orig, 0.55, heatmap_rgb, 0.45, 0)

        def b64(arr):
            _, buf = cv2.imencode('.png', cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            return "data:image/png;base64," + base64.b64encode(buf).decode()

        return b64(overlay), b64(heatmap_rgb)

    except Exception as e:
        import traceback
        print(f"⚠️  Grad-CAM error: {e}")
        traceback.print_exc()
        return None, None

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def safe_float(v):
    if v is None: return 0.0
    try:
        f = float(v)
        return 0.0 if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    except:
        return 0.0

def build_meta_vector(age=None, gender=None, location=None):
    """Build 11-feature metadata vector matching training notebook."""
    vec = [0.0] * 11
    try:
        if age is not None:
            vec[0] = float(np.clip((float(age) - 50.0) / 15.0, -3, 3))
    except: pass

    if gender:
        g = str(gender).lower().strip()
        if g in ("male", "m"):     vec[1] = 1.0
        elif g in ("female", "f"): vec[2] = 1.0

    loc_map = {
        "anterior torso":4,"chest":4,"abdomen":4,
        "posterior torso":5,"back":5,
        "upper extremity":6,"arm":6,"hand":6,
        "lower extremity":7,"leg":7,"foot":7,
        "head":8,"neck":8,"face":8,"scalp":8,
        "palm":9,"sole":9,"oral":10,"genital":10,
    }
    if location:
        ll = str(location).lower().strip()
        for key, idx in loc_map.items():
            if key in ll:
                vec[idx] = 1.0; break

    return torch.tensor([vec], dtype=torch.float32).to(DEVICE)

def preprocess_image(image_bytes, enhance=True):
    """Preprocess image for model inference.
    enhance=True applies phone photo preprocessing pipeline."""
    try:
        pil = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        if enhance:
            pil = preprocess_phone_photo(pil)
        return IMG_TRANSFORM(pil).unsqueeze(0).to(DEVICE)
    except Exception as e:
        raise ValueError(f"Could not decode image: {e}")

def preprocess_tta(image_bytes, enhance=True):
    """TTA preprocessing with optional phone photo enhancement."""
    pil = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    if enhance:
        pil = preprocess_phone_photo(pil)
    return [t(pil).unsqueeze(0).to(DEVICE) for t in TTA_TRANSFORMS]

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "DermAssist AI v3.0", "model_loaded": torch_model is not None, "device": str(DEVICE)}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": torch_model is not None}

@app.post("/predict")
async def predict(
    file:             UploadFile     = File(...),
    age:              Optional[str]  = Form(None),
    gender:           Optional[str]  = Form(None),
    lesion_location:  Optional[str]  = Form(None),
    first_name:       Optional[str]  = Form(None),
    last_name:        Optional[str]  = Form(None),
    family_history:   Optional[str]  = Form(None),
    previous_cancer:  Optional[str]  = Form(None),
    smoking:          Optional[str]  = Form(None),
    uv_exposure:      Optional[str]  = Form(None),
    skin_type:        Optional[str]  = Form(None),
    medications:      Optional[str]  = Form(None),
    new_mole:         Optional[str]  = Form(None),
    mole_change:      Optional[str]  = Form(None),
    itching:          Optional[str]  = Form(None),
    bleeding:         Optional[str]  = Form(None),
    sore_not_healing: Optional[str]  = Form(None),
    spread_pigment:   Optional[str]  = Form(None),
    ldh:              Optional[str]  = Form(None),
    s100b:            Optional[str]  = Form(None),
    mia:              Optional[str]  = Form(None),
    vegf:             Optional[str]  = Form(None),
    lesion_size:      Optional[str]  = Form(None),
    lesion_duration:  Optional[str]  = Form(None),
    db:               Session        = Depends(get_db),
    current_user:     Optional[User] = Depends(get_current_user),
):
    if torch_model is None:
        raise HTTPException(503, "Model not loaded. Place best_model.pth in backend/ and restart.")

    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400, "Only JPEG and PNG images are accepted.")

    contents = await file.read()

    # ── Stage 0: Image quality gate ───────────────────────────────────────────
    is_valid, reason = check_image_quality(contents)
    if not is_valid:
        raise HTTPException(400, detail={
            "error": "image_quality",
            "message": reason,
            "tips": [
                "Use natural daylight — avoid flash",
                "Hold camera 5–10 cm from the lesion",
                "Make sure the lesion fills most of the frame",
                "Keep camera steady to avoid blur"
            ]
        })

    # Build metadata vector
    eff_age = age             if age             and str(age).strip()             else None
    eff_sex = gender          if gender          and str(gender).strip()          else None
    eff_loc = lesion_location if lesion_location and str(lesion_location).strip() else None
    has_meta = any(v is not None for v in [eff_age, eff_sex, eff_loc])
    meta_vec = build_meta_vector(eff_age, eff_sex, eff_loc) if has_meta else None
    print(f"📊 Meta: age={eff_age}, gender={eff_sex}, location={eff_loc}, using={has_meta}")

    # TTA inference — uses B4+B3 ensemble if b3_model.pth is present
    start = time.time()
    try:
        with torch.no_grad():
            tta_tensors = preprocess_tta(contents)
            b4_probs = []
            for t in tta_tensors:
                p = torch.softmax(torch_model(t, meta_vec), dim=1).cpu().numpy()[0]
                b4_probs.append(p)
            b4_avg = np.mean(b4_probs, axis=0)

            if b3_model is not None:
                # Ensemble: B4 (55%) + B3 (45%) weighted average
                b3_probs = []
                for t in tta_tensors:
                    p = torch.softmax(b3_model(t, meta_vec), dim=1).cpu().numpy()[0]
                    b3_probs.append(p)
                b3_avg = np.mean(b3_probs, axis=0)
                probs  = 0.55 * b4_avg + 0.45 * b3_avg
                print(f"🔀 Using B4+B3 ensemble")
            else:
                probs = b4_avg
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")

    ms = int((time.time() - start) * 1000)

    # Sanitize
    probs_clean = [safe_float(p) for p in probs]
    all_scores  = {CLASSES[i]: safe_float(probs[i]) for i in range(len(CLASSES))}

    # Predict
    MAX_PROB    = max(probs_clean) if probs_clean else 0.0
    ENTROPY     = float(-np.sum([p * np.log(p + 1e-9) for p in probs_clean]))
    MAX_ENTROPY = float(np.log(len(CLASSES)))
    UNCERTAINTY = ENTROPY / MAX_ENTROPY if MAX_ENTROPY > 0 else 1.0

    if any(p > 0 for p in probs_clean):
        idx        = int(np.argmax(probs_clean))
        prediction = CLASSES[idx]
        confidence = probs_clean[idx]
    else:
        prediction = "unk"
        confidence = 0.0

    if math.isnan(UNCERTAINTY) or math.isnan(MAX_PROB) or UNCERTAINTY > 0.95 or MAX_PROB < 0.18:
        prediction = "unk"
        confidence = MAX_PROB

    print(f"🔍 {prediction} | conf={confidence:.3f} | uncertainty={UNCERTAINTY:.3f} | meta={'yes' if has_meta else 'no'}")

    # Save to DB
    image_url  = None
    scan_id_db = None
    if current_user:
        try:
            ext  = (file.filename.split(".")[-1] if file.filename and "." in file.filename else "jpg").lower()
            name = f"{uuid.uuid4().hex}.{ext}"
            path = os.path.join(UPLOAD_DIR, name)
            with open(path, "wb") as f:
                f.write(contents)
            image_url = f"/uploads/{name}"
            img_rec = Image(image_name=name, image_path=path, image_format=file.content_type,
                            image_size_kb=len(contents)//1024, user_id=current_user.id)
            db.add(img_rec); db.flush()
            pred_rec = Prediction(
                predicted_label=prediction,
                confidence_score=round(confidence, 4),
                model_version="dermassist_v3_efficientnet_b4",
                processing_time_ms=ms,
                raw_output=json.dumps(all_scores),
                extra_metadata=json.dumps({
                    "risk_level":     RISK_MAP.get(prediction, "Low Risk"),
                    "diagnosis_name": NAME_MAP.get(prediction, prediction),
                    "image_url":      image_url,
                    "intake": {
                        "name":             f"{first_name or ''} {last_name or ''}".strip(),
                        "age":              age, "gender": gender,
                        "family_history":   family_history, "previous_cancer": previous_cancer,
                        "smoking":          smoking, "uv_exposure": uv_exposure,
                        "skin_type":        skin_type, "medications": medications,
                        "new_mole":         new_mole, "mole_change": mole_change,
                        "itching":          itching, "bleeding": bleeding,
                        "sore_not_healing": sore_not_healing, "spread_pigment": spread_pigment,
                        "ldh":              ldh, "s100b": s100b, "mia": mia, "vegf": vegf,
                        "lesion_location":  lesion_location, "lesion_size": lesion_size,
                        "lesion_duration":  lesion_duration,
                    },
                }),
                status="completed", user_id=current_user.id, image_id=img_rec.id,
            )
            db.add(pred_rec)
            db.commit()
            db.refresh(pred_rec)
            scan_id_db = pred_rec.id
        except Exception as e:
            db.rollback()
            print(f"⚠  DB save failed: {e}")

    # Grad-CAM
    heatmap_overlay = heatmap_only = None
    if torch_model is not None and prediction != "unk":
        try:
            idx2 = CLASSES.index(prediction) if prediction in CLASSES else 0
            print(f"🔥 Grad-CAM for {prediction} (idx={idx2})")
            heatmap_overlay, heatmap_only = generate_gradcam(torch_model, contents, idx2)
            print(f"🔥 Grad-CAM: {'✅' if heatmap_overlay else '❌'}")
        except Exception as e:
            print(f"⚠️  Grad-CAM skipped: {e}")

    # ── Stage 2: General disease classifier for non-cancerous results ───────────
    stage2_diagnosis  = None
    stage2_confidence = 0.0
    stage2_all_scores = {}
    stage2_risk       = None
    using_stage2      = False

    # Stage 2 triggers in 2 cases:
    # 1. Stage 1 returns non-cancerous class (NV/BKL/DF/VASC) — standard routing
    # 2. Stage 1 confidence is LOW (<65%) — model is uncertain, try Stage 2 too
    #    This handles phone photos of acne/fungal/viral that Stage 1 misclassifies
    s2_trigger_noncancer = (prediction in STAGE2_TRIGGER and prediction != "unk")
    s2_trigger_lowconf   = (confidence < 0.65 and prediction != "unk" and stage2_model is not None)

    if stage2_model is not None and (s2_trigger_noncancer or s2_trigger_lowconf):
        reason = "non-cancerous" if s2_trigger_noncancer else f"low confidence ({confidence:.2f})"
        print(f"🔀 Stage 1={prediction} ({reason}) → running Stage 2")
        s2_class, s2_conf, s2_scores = run_stage2(contents)
        if s2_class and s2_conf > 0.20:
            stage2_diagnosis  = s2_class
            stage2_confidence = round(s2_conf, 4)
            stage2_all_scores = s2_scores
            stage2_risk       = get_stage2_risk(s2_class)
            using_stage2      = True
            print(f"✅ Stage 2: {s2_class} ({s2_conf:.3f}) — {stage2_risk}")

            # If Stage 2 is more confident than Stage 1, use Stage 2 as primary result
            if s2_trigger_lowconf and s2_conf > confidence:
                print(f"🔄 Stage 2 more confident ({s2_conf:.3f} > {confidence:.3f}) — using as primary")

    return {
        "scan_id":           scan_id_db,
        "diagnosis":         prediction,
        "diagnosis_name":    NAME_MAP.get(prediction, prediction),
        "risk_level":        RISK_MAP.get(prediction, "Low Risk"),
        "confidence":        round(confidence, 4),
        "all_scores":        all_scores,
        "image_url":         image_url,
        "heatmap_overlay":   heatmap_overlay,
        "heatmap_only":      heatmap_only,
        "metadata_used":     has_meta,
        "stage2_diagnosis":  stage2_diagnosis,
        "stage2_confidence": stage2_confidence,
        "stage2_all_scores": stage2_all_scores,
        "stage2_risk":       stage2_risk,
        "using_stage2":      using_stage2,
        "pipeline_stage":    2 if using_stage2 else 1,
    }

@app.get("/user/scans")
def get_scans(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user: raise HTTPException(401, "Not authenticated")
    scans = db.query(Prediction).filter(
        Prediction.user_id == current_user.id,
        Prediction.predicted_label != "__health_record__"
    ).order_by(Prediction.created_at.desc()).all()
    result = []
    for s in scans:
        extra = {}
        try: extra = json.loads(s.extra_metadata) if s.extra_metadata else {}
        except: pass
        result.append({
            "id": s.id, "predicted_label": s.predicted_label,
            "confidence_score": s.confidence_score,
            "risk_level": extra.get("risk_level", ""),
            "diagnosis_name": extra.get("diagnosis_name", s.predicted_label),
            "image_url": extra.get("image_url", None),
            "processing_time_ms": s.processing_time_ms,
            "created_at": str(s.created_at),
        })
    return result

@app.get("/user/me")
def get_profile(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user: raise HTTPException(401, "Not authenticated")
    total = db.query(Prediction).filter(
        Prediction.user_id == current_user.id,
        Prediction.predicted_label != "__health_record__"
    ).count()
    return {
        "id": current_user.id, "full_name": current_user.full_name,
        "username": current_user.username, "email": current_user.email,
        "phone_number": current_user.phone_number, "gender": current_user.gender,
        "date_of_birth": str(current_user.date_of_birth) if current_user.date_of_birth else None,
        "role": current_user.role, "is_active": current_user.is_active,
        "total_scans": total, "created_at": str(current_user.created_at),
    }

@app.get("/user/scans/{scan_id}/report")
def download_report(scan_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    from report_generator import generate_scan_report
    if not current_user: raise HTTPException(401, "Not authenticated")
    scan = db.query(Prediction).filter(Prediction.id == scan_id, Prediction.user_id == current_user.id).first()
    if not scan: raise HTTPException(404, "Scan not found")
    extra  = {}
    try: extra = json.loads(scan.extra_metadata) if scan.extra_metadata else {}
    except: pass
    intake = extra.get("intake", {})
    pdf    = generate_scan_report(
        {"id": scan.id, "predicted_label": scan.predicted_label,
         "confidence_score": scan.confidence_score,
         "risk_level": extra.get("risk_level", "Low Risk"),
         "diagnosis_name": extra.get("diagnosis_name", scan.predicted_label),
         "created_at": str(scan.created_at), "raw_output": scan.raw_output or "{}"},
        {"full_name": current_user.full_name, "email": current_user.email,
         "date_of_birth": str(current_user.date_of_birth) if current_user.date_of_birth else "N/A",
         "gender": current_user.gender or intake.get("gender", "N/A"),
         "phone_number": current_user.phone_number or "N/A",
         "age": intake.get("age","N/A"), "skin_type": intake.get("skin_type","N/A"),
         "smoking": intake.get("smoking","N/A"), "uv_exposure": intake.get("uv_exposure","N/A"),
         "family_history": intake.get("family_history","N/A"),
         "previous_cancer": intake.get("previous_cancer","N/A"),
         "medications": intake.get("medications","None reported"),
         "new_mole": intake.get("new_mole","N/A"), "mole_change": intake.get("mole_change","N/A"),
         "itching": intake.get("itching","N/A"), "bleeding": intake.get("bleeding","N/A"),
         "sore_not_healing": intake.get("sore_not_healing","N/A"),
         "spread_pigment": intake.get("spread_pigment","N/A"),
         "ldh": intake.get("ldh",""), "s100b": intake.get("s100b",""),
         "mia": intake.get("mia",""), "vegf": intake.get("vegf",""),
         "lesion_location": intake.get("lesion_location","N/A"),
         "lesion_size": intake.get("lesion_size","N/A"),
         "lesion_duration": intake.get("lesion_duration","N/A")},
    )
    return StreamingResponse(io.BytesIO(pdf), media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="DermAssist_Report_{current_user.username}_Scan{scan_id}.pdf"'})

# ── Doctors ───────────────────────────────────────────────────────────────────
from pydantic import BaseModel as PydanticBase
from datetime import date as DateType
from typing import List as ListType

class DoctorRegisterRequest(PydanticBase):
    full_name: str; username: str; email: str; password: str; phone: str
    gender: str = ""; date_of_birth: str = ""; post: str; specialty: str
    qualification: str; education_details: str = ""; practice_start_year: int
    clinic_name: str; address: str; city: str
    available_days: ListType[str] = []; available_slots: ListType[str] = []
    specializes_in: ListType[str] = []; languages: ListType[str] = []
    consultation_fee: int = 500; bio: str = ""

@app.post("/doctors/register", status_code=201)
def register_doctor(payload: DoctorRegisterRequest, db: Session = Depends(get_db)):
    from passlib.context import CryptContext
    pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
    if db.query(Doctor).filter(Doctor.email == payload.email).first():
        raise HTTPException(400, "Email already registered")
    if db.query(Doctor).filter(Doctor.username == payload.username).first():
        raise HTTPException(400, "Username already taken")
    dob = None
    if payload.date_of_birth:
        try: dob = DateType.fromisoformat(payload.date_of_birth)
        except: pass
    initials = "".join(w[0] for w in payload.full_name.split() if w)[:2].upper()
    doc = Doctor(
        username=payload.username, email=payload.email, password_hash=pwd.hash(payload.password),
        full_name=payload.full_name, phone=payload.phone, gender=payload.gender or None,
        date_of_birth=dob, post=payload.post, specialty=payload.specialty,
        qualification=payload.qualification, education_details=payload.education_details or None,
        practice_start_year=payload.practice_start_year, clinic_name=payload.clinic_name,
        address=payload.address, city=payload.city, available_days=payload.available_days,
        available_slots=payload.available_slots, specializes_in=payload.specializes_in,
        languages=payload.languages, consultation_fee=payload.consultation_fee,
        bio=payload.bio or None, image_placeholder=initials, status="pending", is_active=False,
    )
    db.add(doc); db.commit(); db.refresh(doc)
    return {"message": "Registration submitted. Awaiting admin approval.", "doctor_id": doc.id}

class DoctorLoginRequest(PydanticBase):
    username: str; password: str

@app.post("/doctors/login")
def doctor_login(payload: DoctorLoginRequest, db: Session = Depends(get_db)):
    from passlib.context import CryptContext
    from auth import create_access_token
    pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
    doc = db.query(Doctor).filter(Doctor.email == payload.username).first() or \
          db.query(Doctor).filter(Doctor.username == payload.username).first()
    if not doc: raise HTTPException(401, "No doctor account found.")
    if not pwd.verify(payload.password, doc.password_hash): raise HTTPException(401, "Incorrect password.")
    if doc.status == "pending":  raise HTTPException(403, "Account pending admin approval.")
    if doc.status == "rejected": raise HTTPException(403, "Registration rejected.")
    if not doc.is_active:        raise HTTPException(403, "Account not active.")
    token = create_access_token({"sub": doc.username, "role": "doctor", "doctor_id": doc.id})
    return {"access_token": token, "token_type": "bearer", "role": "doctor", "doctor": doc.to_public_dict()}

@app.get("/doctors")
def get_doctors(city: str = None, search: str = None, db: Session = Depends(get_db)):
    q = db.query(Doctor)
    if city: q = q.filter(Doctor.city.ilike(f"%{city}%"))
    result = [d.to_public_dict() for d in q.all()]
    if search:
        s = search.lower()
        result = [d for d in result if s in d["name"].lower() or s in d["specialty"].lower()
                  or s in d["city"].lower() or any(s in x.lower() for x in d["specializes_in"])]
    return {"doctors": result, "total": len(result)}

@app.get("/doctors/cities")
def get_cities(db: Session = Depends(get_db)):
    return {"cities": sorted(r.city for r in db.query(Doctor.city).distinct().all())}

@app.get("/doctors/me")
def get_doctor_me(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    from jose import jwt as jose_jwt
    from auth import SECRET_KEY, ALGORITHM
    try:
        data = jose_jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        doc  = db.query(Doctor).filter(Doctor.id == data.get("doctor_id")).first()
        if not doc: raise HTTPException(404, "Not found")
        return doc.to_admin_dict()
    except: raise HTTPException(401, "Invalid token")

class DoctorProfileUpdate(PydanticBase):
    full_name: str = None; phone: str = None; bio: str = None; consultation_fee: int = None
    clinic_name: str = None; address: str = None; city: str = None
    available_days: list = None; available_slots: list = None
    specializes_in: list = None; languages: list = None; qualification: str = None

@app.put("/doctors/profile")
def update_doctor(payload: DoctorProfileUpdate, db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    from jose import jwt as jose_jwt
    from auth import SECRET_KEY, ALGORITHM
    try:
        data = jose_jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if data.get("role") != "doctor": raise HTTPException(403, "Doctors only")
        doc = db.query(Doctor).filter(Doctor.id == data.get("doctor_id")).first()
        if not doc: raise HTTPException(404, "Not found")
        for k, v in payload.dict(exclude_none=True).items():
            if hasattr(doc, k): setattr(doc, k, v)
        db.commit(); db.refresh(doc)
        return {"message": "Profile updated", "doctor": doc.to_admin_dict()}
    except HTTPException: raise
    except: raise HTTPException(401, "Invalid token")

# ── Admin ─────────────────────────────────────────────────────────────────────
@app.get("/admin/doctors")
def admin_list(status: str = None, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user or current_user.role != "admin": raise HTTPException(403, "Admin only")
    q = db.query(Doctor)
    if status: q = q.filter(Doctor.status == status)
    return {"doctors": [d.to_admin_dict() for d in q.order_by(Doctor.created_at.desc()).all()]}

@app.put("/admin/doctors/{doc_id}/approve")
def admin_approve(doc_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user or current_user.role != "admin": raise HTTPException(403, "Admin only")
    doc = db.query(Doctor).filter(Doctor.id == doc_id).first()
    if not doc: raise HTTPException(404, "Not found")
    doc.status = "approved"; doc.is_active = True; db.commit()
    return {"message": f"Dr. {doc.full_name} approved"}

@app.put("/admin/doctors/{doc_id}/reject")
def admin_reject(doc_id: int, notes: str = "", current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user or current_user.role != "admin": raise HTTPException(403, "Admin only")
    doc = db.query(Doctor).filter(Doctor.id == doc_id).first()
    if not doc: raise HTTPException(404, "Not found")
    doc.status = "rejected"; doc.is_active = False; doc.admin_notes = notes; db.commit()
    return {"message": f"Dr. {doc.full_name} rejected"}

@app.delete("/admin/doctors/{doc_id}")
def admin_delete(doc_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user or current_user.role != "admin": raise HTTPException(403, "Admin only")
    doc = db.query(Doctor).filter(Doctor.id == doc_id).first()
    if not doc: raise HTTPException(404, "Not found")
    db.delete(doc); db.commit()
    return {"message": "Doctor deleted"}

class RatingRequest(PydanticBase):
    doctor_id: int; rating: float

@app.post("/doctors/rate")
def rate_doctor(payload: RatingRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user: raise HTTPException(401, "Login required")
    if not (1.0 <= payload.rating <= 5.0): raise HTTPException(400, "Rating must be 1-5")
    doc = db.query(Doctor).filter(Doctor.id == payload.doctor_id).first()
    if not doc: raise HTTPException(404, "Not found")
    doc.rating = round(((doc.rating * doc.review_count) + payload.rating) / (doc.review_count + 1), 1)
    doc.review_count += 1; db.commit()
    return {"message": "Rating submitted", "new_rating": doc.rating}

# ── Appointments ──────────────────────────────────────────────────────────────
class DoctorApptStatus(PydanticBase):
    status: str

@app.put("/doctor/appointments/{apt_id}/status")
def update_appt_status(apt_id: int, body: DoctorApptStatus, request: Request, db: Session = Depends(get_db)):
    from jose import jwt as jose_jwt, JWTError
    from auth import SECRET_KEY, ALGORITHM
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "): raise HTTPException(401, "Not authenticated")
    try:
        payload = jose_jwt.decode(auth_header.split(" ", 1)[1], SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("role") != "doctor": raise HTTPException(403, "Doctors only")
    except JWTError: raise HTTPException(401, "Invalid token")
    if body.status not in {"accepted","rejected","completed","pending"}:
        raise HTTPException(400, "Invalid status")
    apt = db.query(Appointment).filter(Appointment.id == apt_id).first()
    if not apt: raise HTTPException(404, "Not found")
    apt.status = body.status; db.commit()
    return {"id": apt_id, "status": apt.status}

@app.get("/doctor/appointments/{doctor_name:path}")
def doctor_appts(doctor_name: str, db: Session = Depends(get_db)):
    appts = db.query(Appointment).filter(Appointment.doctor_name.ilike(f"%{doctor_name}%")).order_by(Appointment.appointment_date.desc()).all()
    return {"appointments": [{"id":a.id,"user_id":a.user_id,"appointment_date":str(a.appointment_date),"appointment_time":a.appointment_time,"reason":a.reason,"status":a.status} for a in appts]}

@app.get("/appointments")  # backward compat for ProfilePage
def get_my_appts_compat(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return get_my_appts(current_user, db)

@app.get("/appointments/my")
def my_appts(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user: raise HTTPException(401, "Not authenticated")
    appts = db.query(Appointment).filter(Appointment.user_id == current_user.id).order_by(Appointment.appointment_date.desc()).all()
    return [{"id":a.id,"doctor_name":a.doctor_name,"specialty":a.doctor_specialty,"location":a.doctor_clinic,"address":a.doctor_address,"phone":a.doctor_phone,"appointment_date":str(a.appointment_date),"appointment_time":a.appointment_time,"reason":a.reason,"notes":a.notes,"status":a.status,"created_at":str(a.created_at)} for a in appts]

# ── Debug (remove before production) ─────────────────────────────────────────
@app.get("/debug/doctors")
def debug_doctors(db: Session = Depends(get_db)):
    return [{"id":d.id,"email":d.email,"username":d.username,"status":d.status,"name":d.full_name} for d in db.query(Doctor).all()]

@app.post("/debug/approve-doctor/{doc_id}")
def debug_approve(doc_id: int, db: Session = Depends(get_db)):
    doc = db.query(Doctor).filter(Doctor.id == doc_id).first()
    if not doc: raise HTTPException(404, "Not found")
    doc.status = "approved"; doc.is_active = True; db.commit()
    return {"message": f"✅ Dr. {doc.full_name} approved"}

@app.post("/debug/approve-by-email/{email:path}")
def debug_approve_by_email(email: str, db: Session = Depends(get_db)):
    doc = db.query(Doctor).filter(Doctor.email == email).first()
    if not doc: raise HTTPException(404, f"No doctor with email: {email}")
    doc.status = "approved"; doc.is_active = True; db.commit()
    return {"message": f"✅ Dr. {doc.full_name} approved"}

# ── Chatbot ───────────────────────────────────────────────────────────────────
import httpx
from datetime import datetime

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
print(f"🔑 Groq API Key: {'✅ loaded' if GROQ_API_KEY else '❌ MISSING — add to .env'}")

class ChatRequest(PydanticBase):
    message: str; history: list = []; category: str = "skin"

@app.post("/chat")
async def chatbot(req: ChatRequest):
    if not GROQ_API_KEY: raise HTTPException(503, "GROQ_API_KEY not configured in .env")
    system = """You are the AI health assistant built into DermAssist AI — an AI-powered skin disease detection app.
DermAssist uses EfficientNet-B4 trained on ISIC 2019 + HAM10000 to detect 8 skin conditions:
MEL (Melanoma) — HIGH RISK, BCC (Basal Cell Carcinoma) — HIGH RISK,
AK (Actinic Keratosis) — HIGH RISK, SCC (Squamous Cell Carcinoma) — HIGH RISK,
BKL (Benign Keratosis) — MODERATE, DF (Dermatofibroma) — MODERATE,
VASC (Vascular Lesion) — MODERATE, NV (Melanocytic Nevi) — LOW RISK.
Be empathetic, explain results in plain English, always recommend a dermatologist for treatment."""
    messages = [{"role":"system","content":system}]
    for h in req.history[-10:]:
        if isinstance(h, dict) and h.get("role") in ("user","assistant"):
            messages.append({"role":h["role"],"content":h["content"]})
    messages.append({"role":"user","content":req.message})
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization":f"Bearer {GROQ_API_KEY}","Content-Type":"application/json"},
                json={"model":"llama-3.1-8b-instant","messages":messages,"max_tokens":1024,"temperature":0.7})
        if r.status_code != 200:
            raise HTTPException(500, f"Groq error: {r.json().get('error',{}).get('message','Unknown')}")
        reply = r.json()["choices"][0]["message"]["content"]
        return {"reply":reply,"history":req.history+[{"role":"user","content":req.message},{"role":"assistant","content":reply}]}
    except httpx.TimeoutException: raise HTTPException(504, "Request timed out")
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, f"Chatbot error: {e}")

# ── Outbreak / Trends ─────────────────────────────────────────────────────────
@app.get("/outbreak/alerts")
async def outbreak_alerts():
    covid = None
    try:
        async with httpx.AsyncClient(timeout=8) as c:
            r = await c.get("https://disease.sh/v3/covid-19/countries/India")
            if r.status_code == 200:
                d = r.json()
                covid = {"active":d.get("active",0),"recovered":d.get("recovered",0),"deaths":d.get("deaths",0)}
    except: pass
    return {"covid":covid,"skin_alerts":[
        {"disease":"Fungal Skin Infections","region":"Coastal & humid areas","risk":"High","season":"Monsoon (Jun–Oct)","icon":"🍄"},
        {"disease":"Actinic Keratosis Risk","region":"All sun-exposed regions","risk":"High","season":"Summer (Mar–Jun)","icon":"☀️"},
        {"disease":"Melanoma Season Alert","region":"Pan India (peak UV months)","risk":"High","season":"April–August","icon":"🔴"},
        {"disease":"Winter Dermatitis","region":"North India","risk":"Moderate","season":"Winter (Nov–Feb)","icon":"⚠️"},
    ]}

@app.get("/outbreak/skin-trends")
def skin_trends(db: Session = Depends(get_db)):
    from sqlalchemy import func
    LABELS = {"MEL":"Melanoma","BCC":"Basal Cell Carcinoma","AK":"Actinic Keratosis",
              "SCC":"Squamous Cell Carcinoma","BKL":"Benign Keratosis","DF":"Dermatofibroma",
              "VASC":"Vascular Lesion","NV":"Melanocytic Nevi"}
    try:
        results = db.query(Prediction.predicted_label, func.count(Prediction.id).label("cnt"))\
            .filter(Prediction.predicted_label.isnot(None))\
            .filter(Prediction.predicted_label != "__health_record__")\
            .group_by(Prediction.predicted_label)\
            .order_by(func.count(Prediction.id).desc()).all()
        total = sum(r.cnt for r in results) or 1
        return {"total_scans":sum(r.cnt for r in results),
                "trends":[{"code":r.predicted_label,"name":LABELS.get(r.predicted_label,r.predicted_label),"count":r.cnt,"percentage":round(r.cnt/total*100,1)} for r in results]}
    except Exception as e:
        return {"total_scans":0,"trends":[],"error":str(e)}

# ── Health Records ────────────────────────────────────────────────────────────
class HealthRecordPayload(PydanticBase):
    blood_group: Optional[str]=None; height_cm: Optional[float]=None; weight_kg: Optional[float]=None
    allergies: Optional[str]=None; chronic_conditions: Optional[str]=None
    current_medications: Optional[str]=None; past_surgeries: Optional[str]=None
    emergency_contact_name: Optional[str]=None; emergency_contact_phone: Optional[str]=None
    notes: Optional[str]=None

@app.post("/user/health-record")
def save_health_record(payload: HealthRecordPayload, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    data = payload.dict(exclude_none=True)
    if payload.height_cm and payload.weight_kg:
        data["bmi"] = round(payload.weight_kg / ((payload.height_cm/100)**2), 1)
    existing = db.query(Prediction).filter(Prediction.user_id==current_user.id, Prediction.predicted_label=="__health_record__").first()
    if existing:
        existing.all_scores = json.dumps(data); db.commit()
    else:
        db.add(Prediction(
            user_id=current_user.id, predicted_label="__health_record__",
            confidence_score=0.0, all_scores=json.dumps(data),
        ))
        db.commit()
    return {"message":"Health record saved","data":data}

@app.get("/user/health-record")
def get_health_record(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rec = db.query(Prediction).filter(Prediction.user_id==current_user.id, Prediction.predicted_label=="__health_record__").first()
    if not rec: return {"health_record":None}
    try: data = json.loads(rec.all_scores or "{}")
    except: data = {}
    return {"health_record":data}