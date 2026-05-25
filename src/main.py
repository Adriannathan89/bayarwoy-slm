import csv
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from classifier import (
    TransactionClassifier,
    MODEL_PATH,
    FEEDBACK_DATA_PATH,
    VALID_CATEGORIES,
    type_from_category,
)

# --- Config ---
AUTO_REINFORCE_THRESHOLD = 0.85  # confidence minimum untuk auto-save
AUTO_RETRAIN_EVERY = 20          # retrain otomatis setiap N sampel feedback baru

# --- State ---
clf = TransactionClassifier()
_feedback_lock = threading.Lock()
_pending_count = 0
_is_retraining = False


# --- Feedback persistence ---

def _append_feedback(title: str, category: str, source: str):
    FEEDBACK_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not FEEDBACK_DATA_PATH.exists()
    with open(FEEDBACK_DATA_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["title", "category", "source", "timestamp"])
        writer.writerow([title, category, source, datetime.now().isoformat()])


def _background_retrain():
    global _pending_count, _is_retraining
    _is_retraining = True
    try:
        result = clf.retrain_with_feedback()
        print(
            f"[retrain] selesai — base={result['base_samples']}, "
            f"feedback={result['feedback_samples']}, "
            f"total={result['total_samples']}"
        )
        _pending_count = 0
    except Exception as e:
        print(f"[retrain] gagal: {e}")
    finally:
        _is_retraining = False


def maybe_save_and_retrain(title: str, category: str, confidence: float):
    global _pending_count
    if confidence < AUTO_REINFORCE_THRESHOLD:
        return

    with _feedback_lock:
        _append_feedback(title, category, "auto")
        _pending_count += 1
        should_retrain = _pending_count >= AUTO_RETRAIN_EVERY and not _is_retraining

    if should_retrain:
        print(f"[reinforce] {_pending_count} sampel baru → memulai retrain di background...")
        threading.Thread(target=_background_retrain, daemon=True).start()


# --- App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    clf.load(MODEL_PATH)
    print(f"Model dimuat. Kategori: {clf.classes}")
    yield


app = FastAPI(
    title="BayarWoy SLM Service",
    description="Klasifikasi judul transaksi ke kategori pengeluaran (Bahasa Indonesia)",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# --- Schemas ---

class ClassifyRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200, example="makan siang di warteg")


class CategoryResult(BaseModel):
    category: str
    confidence: float


class ClassifyResponse(BaseModel):
    title: str
    category: str
    transaction_type: str  # "pemasukan" atau "pengeluaran"
    confidence: float
    alternatives: list[CategoryResult]
    reinforced: bool


class BatchClassifyRequest(BaseModel):
    titles: list[str] = Field(..., min_length=1, max_length=50)


class BatchClassifyResponse(BaseModel):
    results: list[ClassifyResponse]


class FeedbackRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    correct_category: str = Field(..., example="makanan")


class FeedbackResponse(BaseModel):
    status: str
    message: str
    retrain_triggered: bool


class RetrainResponse(BaseModel):
    status: str
    base_samples: int
    feedback_samples: int
    total_samples: int


# --- Endpoints ---

@app.get("/health")
def health():
    fb_count = 0
    if FEEDBACK_DATA_PATH.exists():
        with open(FEEDBACK_DATA_PATH, encoding="utf-8") as f:
            fb_count = sum(1 for _ in f) - 1  # minus header
    return {
        "status": "ok",
        "model_loaded": clf.pipeline is not None,
        "categories": clf.classes,
        "feedback_samples": max(fb_count, 0),
        "pending_for_retrain": _pending_count,
        "auto_retrain_threshold": AUTO_REINFORCE_THRESHOLD,
        "auto_retrain_every": AUTO_RETRAIN_EVERY,
    }


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    title = req.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Title tidak boleh kosong")
    try:
        result = clf.predict(title)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    reinforced = result["confidence"] >= AUTO_REINFORCE_THRESHOLD
    if reinforced:
        maybe_save_and_retrain(title, result["category"], result["confidence"])

    return ClassifyResponse(
        title=title,
        category=result["category"],
        transaction_type=type_from_category(result["category"]),
        confidence=result["confidence"],
        alternatives=[CategoryResult(**a) for a in result["alternatives"]],
        reinforced=reinforced,
    )


@app.post("/classify/batch", response_model=BatchClassifyResponse)
def classify_batch(req: BatchClassifyRequest):
    results = []
    for title in req.titles:
        title = title.strip()
        if not title:
            continue
        try:
            result = clf.predict(title)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

        reinforced = result["confidence"] >= AUTO_REINFORCE_THRESHOLD
        if reinforced:
            maybe_save_and_retrain(title, result["category"], result["confidence"])

        results.append(ClassifyResponse(
            title=title,
            category=result["category"],
            transaction_type=type_from_category(result["category"]),
            confidence=result["confidence"],
            alternatives=[CategoryResult(**a) for a in result["alternatives"]],
            reinforced=reinforced,
        ))
    return BatchClassifyResponse(results=results)


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    """Koreksi manual: beritahu model kategori yang benar untuk sebuah transaksi."""
    if req.correct_category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=f"Kategori tidak valid. Pilih dari: {sorted(VALID_CATEGORIES)}",
        )
    global _pending_count
    with _feedback_lock:
        _append_feedback(req.title.strip(), req.correct_category, "manual")
        _pending_count += 1
        should_retrain = _pending_count >= AUTO_RETRAIN_EVERY and not _is_retraining

    retrain_triggered = False
    if should_retrain:
        retrain_triggered = True
        threading.Thread(target=_background_retrain, daemon=True).start()

    return FeedbackResponse(
        status="ok",
        message=f"Feedback disimpan: '{req.title}' → {req.correct_category}",
        retrain_triggered=retrain_triggered,
    )


@app.post("/retrain", response_model=RetrainResponse)
def retrain():
    """Trigger retrain manual dengan semua data (base + feedback)."""
    global _is_retraining
    if _is_retraining:
        raise HTTPException(status_code=409, detail="Retrain sedang berjalan.")
    try:
        result = clf.retrain_with_feedback()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return RetrainResponse(status="ok", **result)