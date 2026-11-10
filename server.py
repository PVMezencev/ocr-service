"""
OCR HTTP Server — FastAPI wrapper around EasyOCR and Tesseract
POST /recognize  — upload an image or PDF file, get JSON with recognized text
GET  /health     — health check
"""
import time
import logging
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import easyocr
import torch
import numpy as np
import cv2
import io
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ocr-server")

app = FastAPI(title="OCR Service (EasyOCR + Tesseract)", version="1.1")

_easyocr_reader = None
_tesseract_ready = False

def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        logger.info("Initializing EasyOCR reader (first request, may take time for model download)...")
        t0 = time.time()
        _easyocr_reader = easyocr.Reader(["ru", "en"], gpu=True)
        logger.info(f"EasyOCR reader ready in {time.time()-t0:.1f}s")
    return _easyocr_reader

def get_tesseract():
    """Lazily import pytesseract and verify that ru+eng language data is present."""
    global _tesseract_ready
    if _tesseract_ready:
        import pytesseract
        return pytesseract
    try:
        import pytesseract
    except ImportError as e:
        raise RuntimeError(f"pytesseract not installed: {e}")
    langs = pytesseract.get_languages(config="")
    missing = [l for l in ("rus", "eng") if l not in langs]
    if missing:
        raise RuntimeError(
            f"Tesseract language data missing: {missing}. "
            f"Install with: apt-get install tesseract-ocr-{'-'.join(m)}"
        )
    _tesseract_ready = True
    return pytesseract

def pdf_to_images(data: bytes, dpi: int = 200):
    """Convert PDF pages to list of numpy arrays (BGR).

    Pages with a non-zero /Rotate attribute are derotated via
    page.derotation_matrix, so the rendered pixmap is always upright.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError("PyMuPDF not installed")

    doc = fitz.open(stream=data, filetype="pdf")
    images = []
    for page in doc:
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        if page.rotation % 180 != 0:
            mat = mat * page.derotation_matrix
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        images.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    doc.close()
    return images


def apply_exif_orientation(data: bytes):
    """Decode an image and apply its EXIF orientation tag (if any).

    cv2.imdecode ignores EXIF rotation, so a phone photo taken in portrait
    would be processed sideways. Returns (img, rotated) where rotated is
    True if the pixels were physically rotated.
    """
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, False
    try:
        from PIL import Image, ImageOps
        pil = Image.open(io.BytesIO(data))
        orientation = pil.getexif().get(0x0112, 1)
        if orientation != 1:
            pil = ImageOps.exif_transpose(pil)
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            return img, True
    except Exception as e:
        logger.warning(f"EXIF orientation handling failed: {e}")
    return img, False

def is_pdf(data: bytes) -> bool:
    return data[:4] == b"%PDF"

def recognize_easyocr(reader, img):
    """Run EasyOCR on a BGR image; return list of {text, confidence, bbox}."""
    result = reader.readtext(img)
    entries = []
    for bbox, text, conf in result:
        entries.append({
            "text": text,
            "confidence": round(conf, 4),
            "bbox": [[int(x), int(y)] for x, y in bbox],
        })
    return entries

def recognize_tesseract(pytesseract, img):
    """Run Tesseract on a BGR image; return list of {text, confidence, bbox}."""
    # Tesseract works best on grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(
        gray, lang="rus+eng", config="--oem 3 --psm 3", output_type=pytesseract.Output.DICT
    )
    entries = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        conf = float(data["conf"][i])
        if conf < 0:
            conf = 0.0
        x, y = int(data["left"][i]), int(data["top"][i])
        w, h = int(data["width"][i]), int(data["height"][i])
        entries.append({
            "text": text,
            "confidence": round(conf / 100.0, 4),
            "bbox": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        })
    return entries

# --- Automatic page orientation detection ---------------------------------
# Scanned documents may have pages rotated 90/180/270 degrees (landscape
# content on a portrait page, upside-down scans). We detect the correct
# orientation by running the recognizer on all 4 rotations and picking the
# one with the highest share of high-confidence words.

_CONF_THRESHOLD = 0.80   # word considered "high confidence" if conf >= 0.80
_ORIENTATIONS = (0, 90, 180, 270)
_ROT_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

def _rotate(img, angle):
    if angle == 0:
        return img
    return cv2.rotate(img, _ROT_CODES[angle])

def _orientation_score(entries):
    """Score an orientation: (high_conf_ratio, avg_confidence).

    Primary: share of words with confidence >= _CONF_THRESHOLD.
    Secondary: average confidence (tie-breaker when ratios are close).
    """
    total = len(entries)
    if total == 0:
        return (0.0, 0.0)
    good = sum(1 for e in entries if e["confidence"] >= _CONF_THRESHOLD)
    ratio = good / total
    avg_conf = sum(e["confidence"] for e in entries) / total
    return (ratio, avg_conf)

def auto_orient_easyocr(reader, img):
    """Run EasyOCR on all 4 rotations, return (entries, applied_rotation)."""
    best_angle, best_entries, best_score = 0, None, (-1.0, -1.0)
    for angle in _ORIENTATIONS:
        rotated = _rotate(img, angle)
        entries = recognize_easyocr(reader, rotated)
        score = _orientation_score(entries)
        if score > best_score:
            best_angle, best_entries, best_score = angle, entries, score
        # Early exit: if the original orientation is already excellent,
        # don't bother trying the rest.
        if angle == 0 and score[0] >= 0.85:
            break
    return best_entries, best_angle

def auto_orient_tesseract(pytesseract, img):
    """Run Tesseract on all 4 rotations, return (entries, applied_rotation)."""
    best_angle, best_entries, best_score = 0, None, (-1.0, -1.0)
    for angle in _ORIENTATIONS:
        rotated = _rotate(img, angle)
        entries = recognize_tesseract(pytesseract, rotated)
        score = _orientation_score(entries)
        if score > best_score:
            best_angle, best_entries, best_score = angle, entries, score
        if angle == 0 and score[0] >= 0.85:
            break
    return best_entries, best_angle

@app.get("/health")
def health():
    tesseract_ok = False
    try:
        get_tesseract()
        tesseract_ok = True
    except RuntimeError:
        pass
    return {
        "status": "ok",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "easyocr_loaded": _easyocr_reader is not None,
        "tesseract_available": tesseract_ok,
    }

@app.post("/recognize")
async def recognize(
    file: UploadFile = File(...),
    engine: str = Form("easyocr"),
):
    """Upload an image (jpg/png/bmp/tiff) or PDF and get recognized text.

    engine: "easyocr" (default, GPU, better accuracy) or "tesseract" (CPU, faster).
    """
    if engine not in ("easyocr", "tesseract"):
        return JSONResponse(status_code=400, content={"error": f"Unknown engine: {engine}. Use 'easyocr' or 'tesseract'"})

    data = await file.read()
    if len(data) == 0:
        return JSONResponse(status_code=400, content={"error": "Empty file"})
    if len(data) > 100 * 1024 * 1024:
        return JSONResponse(status_code=413, content={"error": "File too large (max 100MB)"})

    try:
        if engine == "easyocr":
            get_easyocr_reader()
        else:
            get_tesseract()
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    all_results = []
    page_count = 1

    if is_pdf(data):
        try:
            images = pdf_to_images(data)
        except RuntimeError as e:
            return JSONResponse(status_code=500, content={"error": str(e)})
        page_count = len(images)
    else:
        img, _ = apply_exif_orientation(data)
        if img is None:
            return JSONResponse(status_code=400, content={"error": "Cannot decode image. Supported formats: JPG, PNG, BMP, TIFF, PDF"})
        images = [img]

    t0 = time.time()
    orientations_applied = []
    for page_idx, img in enumerate(images):
        if engine == "easyocr":
            reader = get_easyocr_reader()
            page_texts, angle = auto_orient_easyocr(reader, img)
        else:
            pytesseract = get_tesseract()
            page_texts, angle = auto_orient_tesseract(pytesseract, img)
        orientations_applied.append(angle)
        if page_count > 1:
            for entry in page_texts:
                entry["page"] = page_idx + 1
        all_results.extend(page_texts)

    elapsed = time.time() - t0
    full_text = " ".join(t["text"] for t in all_results)

    return {
        "engine": engine,
        "pages": page_count,
        "processing_time_s": round(elapsed, 3),
        "text_count": len(all_results),
        "full_text": full_text,
        "results": all_results,
        "orientations": orientations_applied,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
