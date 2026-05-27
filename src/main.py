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
    ALL_SECONDARY,
    FEEDBACK_DATA_PATH,
    MODEL_PATH,
    TransactionModel,
    VALID_CATEGORIES,
    VALID_SECONDARY_CATEGORIES,
    type_from_category,
)

# --- Config ---
AUTO_REINFORCE_THRESHOLD = 0.85
AUTO_RETRAIN_EVERY       = 100
FEEDBACK_HEADER = ["title", "category", "secondary_category", "source", "timestamp"]

# --- State ---
clf = TransactionModel()
_feedback_lock = threading.Lock()
_pending_count = 0
_is_retraining = False


# --- Feedback persistence ---

def _ensure_feedback_header():
    if not FEEDBACK_DATA_PATH.exists():
        return
    with open(FEEDBACK_DATA_PATH, encoding="utf-8") as f:
        first_line = f.readline().strip()
    if first_line == ",".join(FEEDBACK_HEADER):
        return
    backup = FEEDBACK_DATA_PATH.with_suffix(".csv.bak")
    FEEDBACK_DATA_PATH.rename(backup)
    rows = []
    with open(backup, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "title":              row.get("title", ""),
                "category":           row.get("category", ""),
                "secondary_category": row.get("secondary_category", ""),
                "source":             row.get("source", "unknown"),
                "timestamp":          row.get("timestamp", ""),
            })
    with open(FEEDBACK_DATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEEDBACK_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def _append_feedback(title: str, category: str, secondary: str, source: str):
    FEEDBACK_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not FEEDBACK_DATA_PATH.exists()
    with open(FEEDBACK_DATA_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(FEEDBACK_HEADER)
        writer.writerow([title, category, secondary, source, datetime.now().isoformat()])


def _background_retrain():
    global _pending_count, _is_retraining
    _is_retraining = True
    try:
        result = clf.retrain_with_feedback()
        print(
            f"[retrain] base={result['base_samples']}, "
            f"feedback={result['feedback_samples']}, total={result['total_samples']}"
        )
        _pending_count = 0
    except Exception as e:
        print(f"[retrain] gagal: {e}")
    finally:
        _is_retraining = False


def maybe_save_and_retrain(
    title: str,
    category: str,
    secondary: str,
    confidence: float,
) -> bool:
    global _pending_count
    if confidence < AUTO_REINFORCE_THRESHOLD:
        return False

    with _feedback_lock:
        _append_feedback(title, category, secondary, "auto")
        _pending_count += 1
        should_retrain = _pending_count >= AUTO_RETRAIN_EVERY and not _is_retraining

    if should_retrain:
        print(f"[reinforce] {_pending_count} sampel baru → retrain di background...")
        threading.Thread(target=_background_retrain, daemon=True).start()
    return True


# --- App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    clf.load()
    _ensure_feedback_header()
    print(f"Category  : {len(clf.classes_category)} kelas → {clf.classes_category}")
    print(f"Secondary : {len(clf.classes_secondary)} kelas")
    yield


app = FastAPI(
    title="BayarWoy SLM Service",
    description="Klasifikasi judul transaksi → primary + secondary category (Bahasa Indonesia)",
    version="2.0.0",
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
    secondary_category: str
    transaction_type: str
    confidence: float
    secondary_confidence: float
    alternatives: list[CategoryResult]
    secondary_alternatives: list[CategoryResult]
    reinforced: bool


class BatchClassifyRequest(BaseModel):
    titles: list[str] = Field(..., min_length=1, max_length=50)


class BatchClassifyResponse(BaseModel):
    results: list[ClassifyResponse]


class FeedbackRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    correct_category: str = Field(..., example="makanan")
    correct_secondary_category: str = Field(..., example="jajanan")


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
            fb_count = sum(1 for _ in f) - 1
    return {
        "status":                 "ok",
        "model_loaded":           bool(clf.classes_category),
        "categories":             clf.classes_category,
        "secondary_categories":   clf.classes_secondary,
        "feedback_samples":       max(fb_count, 0),
        "pending_for_retrain":    _pending_count,
        "auto_retrain_threshold": AUTO_REINFORCE_THRESHOLD,
        "auto_retrain_every":     AUTO_RETRAIN_EVERY,
    }


def _classify_one(title: str) -> ClassifyResponse:
    title = title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Title tidak boleh kosong")
    try:
        result = clf.predict(title)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    reinforced = False

    return ClassifyResponse(
        title=title,
        category=result["category"],
        secondary_category=result["secondary_category"],
        transaction_type=result["transaction_type"],
        confidence=result["confidence"],
        secondary_confidence=result["secondary_confidence"],
        alternatives=[CategoryResult(**a) for a in result["alternatives"]],
        secondary_alternatives=[CategoryResult(**a) for a in result["secondary_alternatives"]],
        reinforced=reinforced,
    )


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    return _classify_one(req.title)


@app.post("/classify/batch", response_model=BatchClassifyResponse)
def classify_batch(req: BatchClassifyRequest):
    return BatchClassifyResponse(results=[
        _classify_one(t) for t in req.titles if t.strip()
    ])


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest):
    if req.correct_category not in VALID_CATEGORIES:
        raise HTTPException(
            status_code=422,
            detail=f"Kategori tidak valid. Pilih dari: {sorted(VALID_CATEGORIES)}",
        )
    valid_sec = VALID_SECONDARY_CATEGORIES.get(req.correct_category, set())
    if req.correct_secondary_category not in valid_sec:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Secondary category tidak valid untuk '{req.correct_category}'. "
                f"Pilih dari: {sorted(valid_sec)}"
            ),
        )

    global _pending_count
    with _feedback_lock:
        _append_feedback(
            req.title.strip(),
            req.correct_category,
            req.correct_secondary_category,
            "manual",
        )
        _pending_count += 1
        should_retrain = _pending_count >= AUTO_RETRAIN_EVERY and not _is_retraining

    retrain_triggered = False
    if should_retrain:
        retrain_triggered = True
        threading.Thread(target=_background_retrain, daemon=True).start()

    return FeedbackResponse(
        status="ok",
        message=(
            f"Feedback disimpan: '{req.title}' → "
            f"{req.correct_category}/{req.correct_secondary_category}"
        ),
        retrain_triggered=retrain_triggered,
    )


@app.post("/retrain", response_model=RetrainResponse)
def retrain():
    global _is_retraining
    if _is_retraining:
        raise HTTPException(status_code=409, detail="Retrain sedang berjalan.")
    _is_retraining = True
    try:
        result = clf.retrain_with_feedback()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _is_retraining = False
    return RetrainResponse(
        status="ok",
        base_samples=result["base_samples"],
        feedback_samples=result["feedback_samples"],
        total_samples=result["total_samples"],
    )
