"""
DermAssist AI — Startup Check
Run: python startup_check.py
Tells you exactly which models are loaded and what's missing.
"""
import os

BASE = os.path.dirname(os.path.abspath(__file__))

def check(label, path, required=True):
    exists = os.path.exists(path)
    size   = os.path.getsize(path) / 1024 / 1024 if exists else 0
    status = "✅" if exists else ("❌ REQUIRED" if required else "⚠️  OPTIONAL")
    print(f"  {status} {label}")
    if exists:
        print(f"        Path: {path}")
        print(f"        Size: {size:.1f} MB")
    else:
        print(f"        Expected at: {path}")
    return exists

print("\n" + "="*55)
print("  DermAssist AI — Model & File Status Check")
print("="*55)

print("\n📁 BACKEND MODELS:")
s1 = check("Stage 1 — best_model.pth (B4, 87.2%)",
           os.path.join(BASE, "best_model.pth"), required=True)
s2 = check("Stage 2 — stage2_model.pth (B2, 71.4%)",
           os.path.join(BASE, "stage2_model.pth"), required=False)
s2c = check("Stage 2 — stage2_classes.json",
            os.path.join(BASE, "stage2_classes.json"), required=False)
b3 = check("Ensemble — b3_model.pth (B3, optional)",
           os.path.join(BASE, "b3_model.pth"), required=False)

print("\n📁 BACKEND CONFIG:")
env = check(".env file", os.path.join(BASE, ".env"), required=True)

print("\n📊 PIPELINE STATUS:")
if s1:
    print("  ✅ Stage 1 active — cancer detection working")
else:
    print("  ❌ Stage 1 MISSING — app will not work!")

if s1 and b3:
    print("  ✅ B4+B3 Ensemble active — ~89-90% accuracy")
elif s1:
    print("  ⚠️  B4 only (no ensemble) — ~88% accuracy")

if s2 and s2c:
    print("  ✅ Stage 2 active — general disease detection working")
    try:
        import json
        classes = json.load(open(os.path.join(BASE, "stage2_classes.json")))
        print(f"      Classes ({len(classes)}): {', '.join(classes[:5])}...")
    except: pass
else:
    print("  ⚠️  Stage 2 not found — NV/BKL/DF/VASC will show Stage 1 result only")
    print("      Place stage2_model.pth + stage2_classes.json in backend/ to enable")

print("\n📋 QUICK FIX GUIDE:")
if not s1:
    print("  1. Download best_model.pth from Kaggle Output")
    print("     Place in: backend/best_model.pth")
if not s2:
    print("  2. Download stage2_model.pth from Kaggle Output")
    print("     Place in: backend/stage2_model.pth")
if not s2c:
    print("  3. Download stage2_classes.json from Kaggle Output")
    print("     Place in: backend/stage2_classes.json")
if not b3:
    print("  4. (Optional) Train B3 model using Stage1_B3_Ensemble.ipynb")
    print("     Place b3_model.pth in backend/ for +2-3% accuracy")

print("\n" + "="*55 + "\n")