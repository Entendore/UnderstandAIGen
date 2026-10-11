"""
DeepGuard v21.0 - Multi-Modal AI-Generated Content Detection Suite
=================================================================
Enhanced with LPC Residuals, Q40 Ghost Maps, Background Flicker,
Pronoun Skew, Live Table Filtering, and XML Exports.
"""

import sys
import os
import io
import math
import random
import warnings
import hashlib
import json
import base64
import csv
import re
import sqlite3
import tempfile
import wave
import http.server
import socketserver
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QSize, QRectF, QMimeData, QTimer, QPoint, QRect
from PySide6.QtGui import (
    QAction, QFont, QPixmap, QImage, QPainter, QColor, QPen, QIcon,
    QLinearGradient, QPalette, QDragEnterEvent, QDropEvent, QTextDocument, QKeySequence, QShortcut
)
from PySide6.QtPrintSupport import QPrinter
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QProgressBar, QGroupBox,
    QGridLayout, QFrame, QScrollArea, QSizePolicy, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QMessageBox, QSlider, QSplitter, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QLineEdit, QSystemTrayIcon
)

warnings.filterwarnings("ignore")

# Try optional heavy deps
try:
    import matplotlib
    matplotlib.use("QtAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    from PIL import Image, ImageChops, ImageEnhance, ImageStat
    HAS_PIL = True
except Exception:
    HAS_PIL = False

try:
    from scipy import signal as scipy_signal
    from scipy.fft import fft, fftfreq
    from scipy.stats import kurtosis as scipy_kurtosis, entropy as scipy_entropy, skew as scipy_skew
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False


# ===========================================================================
#  CONFIG PERSISTENCE
# ===========================================================================
CONFIG_PATH = "deepguard_config.json"

def load_config():
    defaults = {
        "sensitivity_audio": 50, "sensitivity_image": 50, 
        "sensitivity_video": 50, "sensitivity_text": 50,
        "custom_lexicon": "", "api_port": 8765, "theme": "dark", "custom_rules": ""
    }
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                defaults.update(json.load(f))
    except: pass
    return defaults

def save_config(config):
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f)
    except: pass


# ===========================================================================
#  STYLING
# ===========================================================================
DARK_QSS = """
QMainWindow { background: #0e1116; }
QTabWidget::pane { border: 1px solid #1f2733; background: #12161d; border-radius: 8px; }
QTabBar::tab { background: #161b22; color: #9aa4b2; padding: 10px 22px; border: 1px solid #1f2733; border-bottom: none; border-top-left-radius: 8px; border-top-right-radius: 8px; font-weight: 600; }
QTabBar::tab:selected { background: #1f6feb; color: white; }
QTabBar::tab:hover:!selected { background: #1c2330; color: #c9d1d9; }
QGroupBox { border: 1px solid #1f2733; border-radius: 8px; margin-top: 14px; padding: 14px; background: #141a22; color: #c9d1d9; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 6px; color: #58a6ff; }
QPushButton { background: #238636; color: white; border: none; padding: 9px 18px; border-radius: 6px; font-weight: 600; }
QPushButton:hover { background: #2ea043; }
QPushButton:pressed { background: #1a7f37; }
QPushButton:disabled { background: #21262d; color: #6e7681; }
QPushButton#secondary { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; }
QPushButton#secondary:hover { background: #2d333b; }
QPushButton#danger { background: #da3633; color: white; }
QPushButton#danger:hover { background: #f85149; }
QTextEdit, QLabel#body { color: #c9d1d9; background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 8px; }
QLabel#title { color: #f0f6fc; font-size: 18px; font-weight: 700; }
QLabel#subtitle { color: #8b949e; font-size: 12px; }
QLabel#metric { color: #58a6ff; font-size: 28px; font-weight: 700; }
QProgressBar { background: #0d1117; border: 1px solid #21262d; border-radius: 6px; text-align: center; color: white; }
QProgressBar::chunk { background: #1f6feb; border-radius: 6px; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background: #0d1117; color: #c9d1d9; border: 1px solid #30363d; padding: 5px; border-radius: 4px; }
QScrollArea { border: none; }
QFrame#card { background: #141a22; border: 1px solid #1f2733; border-radius: 8px; }
QFrame#drop_zone { border: 2px dashed #30363d; border-radius: 8px; background: #0d1117; }
QSlider::groove:horizontal { border: none; height: 4px; background: #30363d; border-radius: 2px; }
QSlider::handle:horizontal { background: #58a6ff; width: 12px; height: 12px; margin: -4px 0; border-radius: 6px; }
QTableWidget { background: #0d1117; color: #c9d1d9; border: 1px solid #21262d; border-radius: 6px; gridline-color: #21262d; }
QHeaderView::section { background: #161b22; color: #58a6ff; padding: 6px; border: none; font-weight: bold; }
QTableWidget::item:selected { background: #1f6feb; color: white; }
"""

LIGHT_QSS = """
QMainWindow { background: #f6f8fa; }
QTabWidget::pane { border: 1px solid #d0d7de; background: #ffffff; border-radius: 8px; }
QTabBar::tab { background: #f6f8fa; color: #57606a; padding: 10px 22px; border: 1px solid #d0d7de; border-bottom: none; border-top-left-radius: 8px; border-top-right-radius: 8px; font-weight: 600; }
QTabBar::tab:selected { background: #ffffff; color: #0969da; }
QTabBar::tab:hover:!selected { background: #eaeef2; color: #24292f; }
QGroupBox { border: 1px solid #d0d7de; border-radius: 8px; margin-top: 14px; padding: 14px; background: #ffffff; color: #24292f; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 6px; color: #0969da; }
QPushButton { background: #2da44e; color: white; border: none; padding: 9px 18px; border-radius: 6px; font-weight: 600; }
QPushButton:hover { background: #2c974b; }
QPushButton:pressed { background: #298e46; }
QPushButton:disabled { background: #eaeef2; color: #8c959f; }
QPushButton#secondary { background: #f6f8fa; color: #24292f; border: 1px solid #d0d7de; }
QPushButton#secondary:hover { background: #eaeef2; }
QPushButton#danger { background: #cf222e; color: white; }
QPushButton#danger:hover { background: #a40e26; }
QTextEdit, QLabel#body { color: #24292f; background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; padding: 8px; }
QLabel#title { color: #24292f; font-size: 18px; font-weight: 700; }
QLabel#subtitle { color: #57606a; font-size: 12px; }
QLabel#metric { color: #0969da; font-size: 28px; font-weight: 700; }
QProgressBar { background: #eaeef2; border: 1px solid #d0d7de; border-radius: 6px; text-align: center; color: #24292f; }
QProgressBar::chunk { background: #0969da; border-radius: 6px; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background: #ffffff; color: #24292f; border: 1px solid #d0d7de; padding: 5px; border-radius: 4px; }
QScrollArea { border: none; }
QFrame#card { background: #ffffff; border: 1px solid #d0d7de; border-radius: 8px; }
QFrame#drop_zone { border: 2px dashed #d0d7de; border-radius: 8px; background: #f6f8fa; }
QSlider::groove:horizontal { border: none; height: 4px; background: #d0d7de; border-radius: 2px; }
QSlider::handle:horizontal { background: #0969da; width: 12px; height: 12px; margin: -4px 0; border-radius: 6px; }
QTableWidget { background: #ffffff; color: #24292f; border: 1px solid #d0d7de; border-radius: 6px; gridline-color: #eaeef2; }
QHeaderView::section { background: #f6f8fa; color: #0969da; padding: 6px; border: none; font-weight: bold; }
QTableWidget::item:selected { background: #0969da; color: white; }
"""

DEFAULT_LLM_LEXICON = {
    "delve", "tapestry", "underscore", "moreover", "furthermore", "testament", 
    "nuance", "intricate", "realm", "landscape", "vibrant", "seamless", 
    "robust", "leverage", "foster", "comprehensive", "multifaceted", 
    "paradigm", "synergy", "evident", "pivotal", "myriad"
}
LLM_TRANSITIONS = {
    "however", "therefore", "moreover", "additionally", "consequently", 
    "furthermore", "thus", "hence", "nevertheless", "accordingly"
}
CAUSAL_CONNECTIVES = {
    "because", "although", "while", "unless", "since", "therefore", 
    "thus", "hence", "consequently", "as", "due", "owing"
}
FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "can", "will", "just", "don", "should", "now", "i", "me", "my", "we", "our",
    "you", "your", "he", "him", "his", "she", "her", "it", "its", "they", "them",
    "their", "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "would", "could", "ought"
}
PRONOUNS = {
    "he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs",
    "it", "its", "i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your", "yours"
}

# ===========================================================================
#  UTILITY HELPERS
# ===========================================================================
def get_file_hash(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(8192)
                if not chunk: break
                h.update(chunk)
        return h.hexdigest()
    except: return "N/A"

def count_syllables(word):
    word = word.lower(); count = 0; vowels = "aeiouy"
    if len(word) == 0: return 0
    if word[0] in vowels: count += 1
    for i in range(1, len(word)):
        if word[i] in vowels and word[i-1] not in vowels: count += 1
    if word.endswith("e"): count -= 1
    return max(1, count)

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def compute_confidence_interval(score_components):
    if not score_components: return 0.0
    n = len(score_components); means = []
    for _ in range(50):
        sample = np.random.choice(score_components, n, replace=True)
        means.append(np.sum(sample))
    return float(np.std(means))

def pixmap_from_pil(pil_img, max_w=400):
    if pil_img.mode != "RGB": pil_img = pil_img.convert("RGB")
    data = pil_img.tobytes("raw", "RGB")
    qimg = QImage(data, pil_img.width, pil_img.height, pil_img.width * 3, QImage.Format_RGB888)
    pm = QPixmap.fromImage(qimg)
    if pm.width() > max_w: pm = pm.scaledToWidth(max_w, Qt.SmoothTransformation)
    return pm

def calculate_mtld(tokens):
    def calc_sub(toks):
        types = set(); n = 0
        for i, token in enumerate(toks, 1):
            types.add(token)
            ttr = len(types) / i
            if ttr < 0.72: return i / (1 - ttr)
            n = i
        return n / (1 - (len(types) / n)) if n > 0 else 0
    if not tokens: return 0.0
    return (calc_sub(tokens) + calc_sub(tokens[::-1])) / 2

def fast_lbp_variance(patch_u8):
    if patch_u8.shape[0] < 3 or patch_u8.shape[1] < 3: return 0.0
    center = patch_u8[1:-1, 1:-1]
    mask = np.zeros_like(center)
    mask |= (patch_u8[0:-2, 0:-2] > center) << 0
    mask |= (patch_u8[0:-2, 1:-1] > center) << 1
    mask |= (patch_u8[0:-2, 2:] > center) << 2
    mask |= (patch_u8[1:-1, 2:] > center) << 3
    mask |= (patch_u8[2:, 2:] > center) << 4
    mask |= (patch_u8[2:, 1:-1] > center) << 5
    mask |= (patch_u8[2:, 0:-2] > center) << 6
    mask |= (patch_u8[1:-1, 0:-2] > center) << 7
    hist, _ = np.histogram(mask, bins=256, range=(0, 256))
    hist = hist / (hist.sum() + 1e-9)
    return float(np.var(hist))

def generate_chain_of_custody_hash(res):
    payload = {
        "file_name": res.get("file_name", "N/A"),
        "file_hash": res.get("file_hash", "N/A"),
        "score": res.get("score", 0.0),
        "verdict": res.get("verdict", "N/A"),
        "metrics": res.get("metrics", {}),
        "reasons": res.get("reasons", [])
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def parse_custom_rules(rules_str):
    rules = []
    if not rules_str: return rules
    try:
        for part in rules_str.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                rules.append({"key": k.strip().lower(), "val": float(v.strip())})
    except: pass
    return rules

def apply_custom_rules(text, rules):
    score_mod = 0.0
    matched = []
    text_lower = text.lower()
    for rule in rules:
        if rule["key"] in text_lower:
            score_mod += rule["val"]
            matched.append(f"Custom rule '{rule['key']}' ({rule['val']:+.1f}%)")
    return score_mod / 100.0, matched

def dict_to_xml(tag, d):
    elem = ET.Element(tag)
    for key, val in d.items():
        child = ET.SubElement(elem, key)
        if isinstance(val, dict):
            child.append(dict_to_xml('data', val))
        elif isinstance(val, list):
            for item in val:
                li = ET.SubElement(child, 'item')
                li.text = str(item)
        else:
            child.text = str(val)
    return elem


# ===========================================================================
#  DETECTION WORKER THREAD
# ===========================================================================
class DetectionWorker(QThread):
    progress = Signal(int, str)
    finished_result = Signal(dict)

    def __init__(self, modality, file_path, sensitivity, deep_scan=False, custom_lexicon=None, custom_rules="", api_mode=False):
        super().__init__()
        self.modality, self.file_path = modality, file_path
        self.sensitivity, self.deep_scan = sensitivity, deep_scan
        self.custom_lexicon = custom_lexicon
        self.custom_rules = custom_rules
        self.api_mode = api_mode

    def run(self):
        try:
            self.progress.emit(5, f"Computing file hash...")
            file_hash = get_file_hash(self.file_path)
            self.progress.emit(10, f"Loading {self.modality} file...")
            
            if self.modality == "audio": res = detect_audio(self.file_path, self.progress, self.deep_scan, self.api_mode)
            elif self.modality == "image": res = detect_image(self.file_path, self.progress, self.deep_scan, self.api_mode)
            elif self.modality == "video": res = detect_video(self.file_path, self.progress, self.deep_scan, self.api_mode)
            elif self.modality == "text": res = detect_text(self.file_path, self.progress, self.deep_scan, self.custom_lexicon, self.custom_rules, self.api_mode)
            else: res = {"error": "Unknown modality"}
            
            res["file_hash"] = file_hash; res["sensitivity"] = self.sensitivity
            res["file_name"] = os.path.basename(self.file_path); res["modality"] = self.modality
            
            threshold = 0.50 + (self.sensitivity - 50) / 166.0 
            if "score" in res:
                res["threshold"] = threshold
                res["verdict"] = res.get("verdict_fake", "AI-Generated") if res["score"] > threshold else res.get("verdict_real", "Likely Authentic")
                res["margin_of_error"] = compute_confidence_interval(res.get("score_components", []))

            self.progress.emit(100, "Done")
            self.finished_result.emit(res)
        except Exception as e:
            self.finished_result.emit({"error": str(e)})


# ===========================================================================
#  AUDIO DETECTION (Upgraded with LPC Residual Energy Variance)
# ===========================================================================
def detect_audio(path, progress_cb, deep_scan, api_mode=False):
    try: wf = wave.open(path, "rb")
    except: return {"error": "Only WAV supported. Convert via ffmpeg."}

    n_channels, sampwidth, framerate, n_frames = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
    raw = wf.readframes(n_frames); wf.close()

    if sampwidth == 2: data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    elif sampwidth == 1: data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128
    elif sampwidth == 4: data = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
    else: return {"error": f"Unsupported sample width: {sampwidth}"}

    stereo_corr = 0.0
    if n_channels == 2 and sampwidth > 1:
        try:
            if sampwidth == 2: stereo_data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            elif sampwidth == 4: stereo_data = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
            else: stereo_data = np.frombuffer(raw, dtype=np.float32)
            stereo_data = stereo_data.reshape(-1, 2)
            if len(stereo_data) > 100:
                left = stereo_data[:, 0]; right = stereo_data[:, 1]
                stereo_corr = float(np.corrcoef(left, right)[0, 1])
        except: pass

    if n_channels > 1: data = data.reshape(-1, n_channels).mean(axis=1)
    data = data / (np.max(np.abs(data)) + 1e-9)
    duration = len(data) / framerate

    fft_size = 1 << 19 if deep_scan else 1 << 18
    N = min(len(data), fft_size)
    seg = data[:N] * np.hanning(N)
    spectrum = np.abs(fft(seg))[:N // 2]
    freqs = fftfreq(N, 1 / framerate)[:N // 2]

    spectral_flatness = np.exp(np.mean(np.log(spectrum + 1e-9))) / (np.mean(spectrum) + 1e-9)
    hf_ratio = np.sum(spectrum[freqs > 4000] ** 2) / (np.sum(spectrum ** 2) + 1e-9)
    
    cepstrum = np.abs(np.fft.ifft(np.log(spectrum + 1e-9)))
    cpp_global = np.max(cepstrum[20:200]) if len(cepstrum) > 200 else 0

    block = framerate // 4
    zcr_blocks, rms_blocks = [], []
    for i in range(0, len(data) - block, block):
        seg = data[i:i + block]
        zcr_blocks.append(np.mean(np.abs(np.diff(np.sign(seg))) > 0))
        rms_blocks.append(np.sqrt(np.mean(seg**2)))
    zcr_std = np.std(zcr_blocks) if zcr_blocks else 0
    dyn_range_var = float(np.var(rms_blocks)) if rms_blocks else 0.0
    rms_grad = np.abs(np.diff(rms_blocks))
    breath_var = float(np.var(rms_grad)) if len(rms_grad) > 0 else 0.0

    reverb_decay_var = 0.0
    if len(rms_blocks) > 10 and HAS_SCIPY:
        peaks = scipy_signal.find_peaks(rms_blocks, height=np.mean(rms_blocks))[0]
        decays = []
        for p in peaks:
            if p + 5 < len(rms_blocks):
                decay_curve = rms_blocks[p:p+5]
                if decay_curve[0] > 0: decays.append((decay_curve[-1] - decay_curve[0]) / decay_curve[0])
        if len(decays) > 0: reverb_decay_var = float(np.var(decays))

    f0_contour = []
    frame_len = int(0.03 * framerate)
    for i in range(0, len(data) - frame_len, frame_len):
        frame = data[i:i+frame_len]
        if np.max(np.abs(frame)) < 0.01: continue
        corr = np.correlate(frame, frame, mode='full')[frame_len:]
        lag_min, lag_max = int(framerate / 500), int(framerate / 80)
        if lag_max > len(corr): lag_max = len(corr)
        if lag_min < lag_max:
            f0_contour.append(framerate / (np.argmax(corr[lag_min:lag_max]) + lag_min))
            
    f0_std = np.std(f0_contour) if f0_contour else 0
    f0_mean = np.mean(f0_contour) if f0_contour else 0
    f0_jitter = float(np.mean(np.abs(np.diff(f0_contour))) / (f0_mean + 1e-9)) if len(f0_contour) > 2 else 0.0

    formant_bw_var = 0.0
    lpc_resid_var = 0.0
    if HAS_SCIPY and len(data) > framerate and not api_mode:
        lpc_seg = data[frame_len:frame_len*2] * np.hanning(frame_len)
        if np.max(np.abs(lpc_seg)) > 0.01:
            order = int(2 * framerate / 1000) + 2
            a = scipy_signal.lpc(lpc_seg, order=order)
            roots = np.roots(a)
            roots = roots[np.imag(roots) > 0]
            angz = np.arctan2(np.imag(roots), np.real(roots))
            frqs = angz * (framerate / (2 * np.pi))
            bws = -0.5 * (framerate / (2 * np.pi)) * np.log(np.abs(roots))
            valid_bws = [b for f, b in zip(frqs, bws) if 90 < f < 4000]
            if valid_bws: formant_bw_var = float(np.var(valid_bws))
            
            # NEW: LPC Residual Energy Variance
            # Real speech has chaotic noise in the residual. TTS is clean.
            residual = scipy_signal.lfilter(a, [1], lpc_seg)
            lpc_resid_var = float(np.var(residual))

    rolloff_var = 0.0
    phase_cont_var = 0.0
    phase_coherence_var = 0.0
    hnr_var = 0.0
    cpp_var = 0.0
    if HAS_SCIPY and not api_mode:
        f, t, Zxx = scipy_signal.stft(data[:framerate * 4], framerate, nperseg=1024)
        magnitudes = np.abs(Zxx)
        roll_offs = []
        for i in range(magnitudes.shape[1]):
            mag = magnitudes[:, i]
            total_energy = np.sum(mag**2)
            cum_energy = np.cumsum(mag**2)
            idx = np.searchsorted(cum_energy, 0.85 * total_energy)
            if idx < len(f): roll_offs.append(f[idx])
        if len(roll_offs) > 0: rolloff_var = float(np.var(roll_offs))

        phases = np.unwrap(np.angle(Zxx), axis=1)
        phase_cont_var = float(np.var(np.diff(phases, axis=1)))
        
        vocal_mask = (f > 100) & (f < 4000)
        if np.any(vocal_mask):
            phase_diffs = np.diff(phases[vocal_mask, :], axis=0)
            phase_coherence_var = float(np.var(phase_diffs))
            
        hnrs = []
        cpps = []
        for i in range(magnitudes.shape[1]):
            mag = magnitudes[:, i]
            if len(f) > 10:
                harmonic_e = np.max(mag)**2
                noise_e = np.median(mag)**2
                if noise_e > 0: hnrs.append(10 * np.log10(harmonic_e / noise_e))
                
                log_mag = np.log(mag + 1e-9)
                c = np.abs(np.fft.ifft(log_mag))
                if len(c) > 50:
                    cpps.append(np.max(c[10:50]))
        if len(hnrs) > 0: hnr_var = float(np.var(hnrs))
        if len(cpps) > 0: cpp_var = float(np.var(cpps))

    band_entropy = 0.0
    if HAS_SCIPY:
        bands = np.array_split(spectrum, 10)
        band_energies = np.array([np.sum(b**2) for b in bands])
        band_energies = band_energies / (band_energies.sum() + 1e-9)
        band_entropy = float(scipy_entropy(band_energies))

    noise_flatness = 0.0
    if rms_blocks:
        silence_thresh = np.percentile(rms_blocks, 10)
        silent_segments = [i for i, r in enumerate(rms_blocks) if r < silence_thresh]
        if silent_segments:
            silent_data = np.concatenate([data[i*block : (i+1)*block] for i in silent_segments])
            if len(silent_data) > fft_size:
                silent_spec = np.abs(fft(silent_data[:fft_size] * np.hanning(fft_size)))[:fft_size // 2]
                noise_flatness = np.exp(np.mean(np.log(silent_spec + 1e-9))) / (np.mean(silent_spec) + 1e-9)

    flux_std = 0.0
    if HAS_SCIPY and not api_mode:
        f, t, Zxx = scipy_signal.stft(data[:framerate * 4], framerate, nperseg=512)
        flux_std = np.std(np.linalg.norm(np.diff(np.abs(Zxx), axis=1), axis=0))

    score, reasons, radar, score_components = 0.0, [], {"HF Energy": 0, "F0 Smooth": 0, "Jitter": 0, "HNR Var": 0, "LPC Res": 0}, []
    if spectral_flatness < 0.05: s=0.2; score+=s; reasons.append("Unnaturally low spectral flatness"); score_components.append(s)
    elif spectral_flatness > 0.4: s=0.1; score+=s; reasons.append("High spectral flatness (GAN artifact)"); score_components.append(s)
    if hf_ratio < 0.005: s=0.2; score+=s; reasons.append(f"Low HF energy"); radar["HF Energy"]=0.9; score_components.append(s)
    if f0_std < 10.0 and f0_mean > 50: s=0.25; score+=s; reasons.append(f"F0 pitch contour too smooth"); radar["F0 Smooth"]=0.9; score_components.append(s)
    if f0_jitter < 0.02 and f0_mean > 50: s=0.3; score+=s; reasons.append(f"Near-zero pitch jitter"); radar["Jitter"]=0.95; score_components.append(s)
    if rolloff_var < 10000.0 and rolloff_var > 0.0: s=0.25; score+=s; reasons.append(f"Low spectral roll-off variance"); score_components.append(s)
    if phase_cont_var < 1.0 and phase_cont_var > 0.0: s=0.25; score+=s; reasons.append(f"Unnatural phase continuity variance"); score_components.append(s)
    if phase_coherence_var < 1.0 and phase_coherence_var > 0.0: s=0.3; score+=s; reasons.append(f"Low harmonic phase coherence variance"); score_components.append(s)
    if hnr_var < 5.0 and hnr_var > 0.0: s=0.25; score+=s; reasons.append(f"Low HNR variance"); radar["HNR Var"]=0.9; score_components.append(s)
    if cpp_var < 0.5 and cpp_var > 0.0: s=0.25; score+=s; reasons.append(f"Low Cepstral Peak Prominence variance"); score_components.append(s)
    
    # TTS cleans up vocal tract residual noise
    if lpc_resid_var < 0.001 and lpc_resid_var > 0.0:
        s=0.25; score+=s; reasons.append(f"Low LPC residual variance ({lpc_resid_var:.4f}) - smoothed vocal tract"); radar["LPC Res"]=0.9; score_components.append(s)

    if band_entropy < 1.5 and HAS_SCIPY: s=0.2; score+=s; reasons.append(f"Low frequency band entropy"); score_components.append(s)
    if noise_flatness > 0.3 or noise_flatness == 0.0: s=0.15; score+=s; reasons.append("Unnatural noise floor"); score_components.append(s)
    if flux_std < 1.0 and HAS_SCIPY and not api_mode: s=0.2; score+=s; reasons.append(f"Low STFT Spectral Flux variance"); score_components.append(s)
    if dyn_range_var < 0.005: s=0.2; score+=s; reasons.append(f"Low dynamic range variance"); score_components.append(s)
    if formant_bw_var < 100.0 and formant_bw_var > 0.0: s=0.25; score+=s; reasons.append(f"Low formant bandwidth variance"); score_components.append(s)
    if breath_var < 0.0005 and len(rms_blocks) > 10: s=0.2; score+=s; reasons.append(f"Unnatural breath/silence transitions"); score_components.append(s)
    if reverb_decay_var < 0.005 and len(rms_blocks) > 10: s=0.2; score+=s; reasons.append(f"Unnatural reverb decay variance"); score_components.append(s)
    if n_channels == 2 and (abs(stereo_corr) > 0.99 or stereo_corr < -0.95): s=0.2; score+=s; reasons.append(f"Suspicious stereo correlation"); score_components.append(s)

    score = float(min(0.95, score + 0.1))
    
    viz = None
    if HAS_MPL and not api_mode:
        fig = Figure(figsize=(8, 6), facecolor="#0d1117" if QApplication.instance().property("Theme") == "dark" else "#ffffff")
        ax1 = fig.add_subplot(221); ax1.specgram(data[:framerate * 4], Fs=framerate, NFFT=1024, noverlap=512, cmap='viridis')
        ax1.set_title("Spectrogram", color="#c9d1d9" if QApplication.instance().property("Theme") == "dark" else "#24292f", fontsize=9)
        ax1.tick_params(colors="#8b949e" if QApplication.instance().property("Theme") == "dark" else "#57606a", labelsize=7)
        for spine in ax1.spines.values(): spine.set_color("#30363d" if QApplication.instance().property("Theme") == "dark" else "#d0d7de")
        ax2 = fig.add_subplot(222); ax2.plot(f0_contour, color="#f85149", linewidth=1.2)
        ax2.set_title("F0 Pitch Contour", color="#c9d1d9" if QApplication.instance().property("Theme") == "dark" else "#24292f", fontsize=9)
        ax2.tick_params(colors="#8b949e" if QApplication.instance().property("Theme") == "dark" else "#57606a", labelsize=7)
        for spine in ax2.spines.values(): spine.set_color("#30363d" if QApplication.instance().property("Theme") == "dark" else "#d0d7de")
        ax3 = fig.add_subplot(223, polar=True)
        labels, stats = list(radar.keys()), list(radar.values())
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        stats += stats[:1]; angles += angles[:1]
        ax3.plot(angles, stats, color="#58a6ff", linewidth=2); ax3.fill(angles, stats, color="#58a6ff", alpha=0.3)
        ax3.set_xticks(angles[:-1]); ax3.set_xticklabels(labels, color="#c9d1d9" if QApplication.instance().property("Theme") == "dark" else "#24292f", fontsize=8)
        ax3.set_title("Anomaly Radar", color="#c9d1d9" if QApplication.instance().property("Theme") == "dark" else "#24292f", fontsize=9, pad=15)
        ax3.grid(color="#30363d")
        ax4 = fig.add_subplot(224); ax4.semilogx(freqs[1:], 20*np.log10(spectrum[1:] + 1e-9), color="#3fb950", linewidth=0.6)
        ax4.set_title("Magnitude Spectrum (dB)", color="#c9d1d9" if QApplication.instance().property("Theme") == "dark" else "#24292f", fontsize=9)
        ax4.tick_params(colors="#8b949e" if QApplication.instance().property("Theme") == "dark" else "#57606a", labelsize=7)
        for spine in ax4.spines.values(): spine.set_color("#30363d" if QApplication.instance().property("Theme") == "dark" else "#d0d7de")
        fig.tight_layout(); viz = fig

    return {
        "score": score, "score_components": score_components, "radar": radar,
        "verdict_fake": "AI-Generated / Synthetic", "verdict_real": "Likely Authentic",
        "metrics": {
            "Duration (s)": f"{duration:.2f}", "Spectral Flatness": f"{spectral_flatness:.4f}",
            "HF Energy Ratio": f"{hf_ratio*100:.3f}%", "Pitch Jitter": f"{f0_jitter:.4f}",
            "HNR Variance": f"{hnr_var:.2f}", "LPC Resid Var": f"{lpc_resid_var:.4f}",
            "Stereo Corr": f"{stereo_corr:.2f}" if n_channels == 2 else "Mono"
        },
        "reasons": reasons, "viz": viz,
        "methodology": "Inspired by AASIST (Interspeech 2022). Adds LPC Residual Energy Variance. Real human vocal tracts produce chaotic noise during unvoiced consonants; TTS models smooth this out."
    }


# ===========================================================================
#  IMAGE DETECTION (Upgraded with Secondary JPEG Ghost Map Q40)
# ===========================================================================
def detect_image(path, progress_cb, deep_scan, api_mode=False):
    if not HAS_PIL: return {"error": "Pillow required for image analysis."}
    img = Image.open(path).convert("RGB"); W, H = img.size
    arr = np.array(img).astype(np.float32); gray = arr.mean(axis=2)

    F = np.fft.fftshift(np.fft.fft2(gray)); mag = np.abs(F)
    cy, cx = np.array(gray.shape) // 2
    Y, X = np.ogrid[:gray.shape[0], :gray.shape[1]]
    r = np.sqrt((Y - cy)**2 + (X - cx)**2).astype(int); rmax = min(gray.shape) // 2
    hf_ratio = np.sum(mag[r > rmax // 4]**2) / (np.sum(mag**2) + 1e-9)
    fft_peak_ratio = float(np.max(mag) / (np.mean(mag) + 1e-9))

    grid_size = 8 if deep_scan else 4
    patch_h, patch_w = H // grid_size, W // grid_size
    patch_hfs = []
    for i in range(grid_size):
        for j in range(grid_size):
            patch = gray[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w]
            if patch.shape[0] < 10 or patch.shape[1] < 10: continue
            F_p = np.fft.fftshift(np.fft.fft2(patch)); mag_p = np.abs(F_p)
            cy_p, cx_p = np.array(patch.shape) // 2
            Y_p, X_p = np.ogrid[:patch.shape[0], :patch.shape[1]]
            r_p = np.sqrt((Y_p - cy_p)**2 + (X_p - cx_p)**2).astype(int); rmax_p = min(patch.shape) // 2
            pe = np.sum(mag_p**2)
            if pe > 0: patch_hfs.append(np.sum(mag_p[r_p > rmax_p // 4]**2) / pe)
    patch_fft_var = float(np.std(patch_hfs)) if patch_hfs else 0

    noise_kurtosis, edge_grad_var, sat_entropy, prnu_corr, ca_var, dof_var, vignette_ratio, nyquist_energy = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0
    if HAS_CV2 and HAS_SCIPY:
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_kurtosis = float(scipy_kurtosis(laplacian.flatten(), fisher=True))
        
        F_lap = np.fft.fftshift(np.fft.fft2(laplacian))
        nyquist_mask = r > (rmax * 0.9)
        nyquist_energy = float(np.sum(F_lap[nyquist_mask]**2) / (np.sum(F_lap**2) + 1e-9))
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5); sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        edge_grad_var = float(np.var(grad_mag))
        
        block_size = 64; h, w = laplacian.shape
        sharpness_map = np.zeros((h // block_size, w // block_size))
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                sharpness_map[i // block_size, j // block_size] = np.var(laplacian[i:i+block_size, j:j+block_size])
        dof_var = float(np.var(sharpness_map))

        hsv = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2HSV)
        sat_hist = np.histogram(hsv[:,:,1].flatten(), bins=32, range=(0, 256))[0]
        sat_entropy = float(scipy_entropy(sat_hist / (sat_hist.sum() + 1e-9)))
        
        h, w = laplacian.shape
        left_noise = laplacian[:, :w//2].flatten(); right_noise = laplacian[:, w//2:].flatten()
        left_noise = (left_noise - np.mean(left_noise)) / (np.std(left_noise) + 1e-9)
        right_noise = (right_noise - np.mean(right_noise)) / (np.std(right_noise) + 1e-9)
        prnu_corr = float(np.corrcoef(left_noise, right_noise)[0, 1])

        edges = cv2.Canny(gray.astype(np.uint8), 100, 200)
        r_channel = arr[:,:,0]; b_channel = arr[:,:,2]
        rb_diff = np.abs(r_channel - b_channel)
        if np.sum(edges > 0) > 0: ca_var = float(np.var(rb_diff[edges > 0]))
        
        ch, cw = H//8, W//8
        if ch > 0 and cw > 0:
            center_brightness = np.mean(gray[ch:3*ch, cw:3*cw])
            corners = [
                gray[:ch, :cw], gray[:ch, -cw:], 
                gray[-ch:, :cw], gray[-ch:, -cw:]
            ]
            corner_brightness = np.mean(corners)
            if center_brightness > 0:
                vignette_ratio = float(corner_brightness / center_brightness)

    buf90 = io.BytesIO(); img.save(buf90, "JPEG", quality=90); buf90.seek(0)
    re90 = np.array(Image.open(buf90).convert("RGB")).astype(np.float32)
    ela_diff = np.abs(arr - re90); ela_mean = ela_diff.mean()

    ghost_var = 0.0
    ghost_q40_var = 0.0
    if not api_mode:
        buf50 = io.BytesIO(); img.save(buf50, "JPEG", quality=50); buf50.seek(0)
        re50 = np.array(Image.open(buf50).convert("RGB")).astype(np.float32)
        ghost_diff = np.abs(re90 - re50)
        block_size = 32; h, w = ghost_diff.shape[:2]
        ghost_heatmap = np.zeros((h // block_size, w // block_size))
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                ghost_heatmap[i // block_size, j // block_size] = ghost_diff[i:i+block_size, j:j+block_size].mean()
        ghost_var = float(np.std(ghost_heatmap) / (np.mean(ghost_heatmap) + 1e-9))
        
        # NEW: Secondary JPEG Ghost Map (Q40 vs Q90)
        # Spliced regions often reveal themselves only at extreme compression differences
        buf40 = io.BytesIO(); img.save(buf40, "JPEG", quality=40); buf40.seek(0)
        re40 = np.array(Image.open(buf40).convert("RGB")).astype(np.float32)
        ghost_diff_40 = np.abs(re90 - re40)
        ghost_heatmap_40 = np.zeros((h // block_size, w // block_size))
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                ghost_heatmap_40[i // block_size, j // block_size] = ghost_diff_40[i:i+block_size, j:j+block_size].mean()
        ghost_q40_var = float(np.std(ghost_heatmap_40) / (np.mean(ghost_heatmap_40) + 1e-9))

    block_size = 32; h, w = ela_diff.shape[:2]
    ela_heatmap = np.zeros((h // block_size, w // block_size))
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            ela_heatmap[i // block_size, j // block_size] = ela_diff[i:i+block_size, j:j+block_size].mean()
    block_var = np.std(ela_heatmap) / (np.mean(ela_heatmap) + 1e-9)

    face_ela_score, has_face, faces, eye_asymmetry, light_asymmetry, eye_reflection_dist = 0.0, False, [], 0.0, 0.0, 0.0
    if HAS_CV2:
        cv2_img = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_cv2 = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_cv2, 1.3, 5)
        if len(faces) > 0:
            has_face = True; face_elas = []
            for (x, y, fw, fh) in faces:
                face_elas.append(ela_diff[y:y+fh, x:x+fw].mean())
                eye_y = int(y + fh * 0.25); eye_h = int(fh * 0.15)
                left_eye = gray[eye_y:eye_y+eye_h, x:int(x+fw*0.45)]
                right_eye = gray[eye_y:eye_y+eye_h, int(x+fw*0.55):x+fw]
                left_spec = np.mean(left_eye > np.percentile(left_eye, 95))
                right_spec = np.mean(right_eye > np.percentile(right_eye, 95))
                eye_asymmetry = float(abs(left_spec - right_spec))
                
                l_y, l_x = np.unravel_index(np.argmax(left_eye), left_eye.shape)
                r_y, r_x = np.unravel_index(np.argmax(right_eye), right_eye.shape)
                eye_reflection_dist = float(abs(l_y - r_y) + abs(l_x - r_x)) / fw
                
                face_left = gray[y:y+fh, x:int(x+fw*0.5)]
                face_right = gray[y:y+fh, int(x+fw*0.5):x+fw]
                light_asymmetry = float(abs(np.mean(face_left) - np.mean(face_right)))
            face_ela_score = np.mean(face_elas)
            if abs(face_ela_score - ela_mean) > 2.0: block_var += 0.5 

    score, reasons, radar, score_components = 0.0, [], {"HF Energy": 0, "Patch Var": 0, "PRNU": 0, "ELA Var": 0, "Ghost Q40": 0}, []
    if hf_ratio < 0.02: s=0.15; score+=s; reasons.append(f"Low global HF energy"); radar["HF Energy"]=0.8; score_components.append(s)
    if patch_fft_var < 0.01: s=0.2; score+=s; reasons.append(f"Suspiciously uniform Patch-FFT variance"); radar["Patch Var"]=0.9; score_components.append(s)
    if noise_kurtosis > 5.0 or (noise_kurtosis < 2.0 and noise_kurtosis > 0): s=0.25; score+=s; reasons.append(f"Unnatural sensor noise kurtosis"); score_components.append(s)
    if abs(prnu_corr) < 0.1: s=0.3; score+=s; reasons.append(f"Low PRNU noise cross-correlation - splicing"); radar["PRNU"]=0.9; score_components.append(s)
    if edge_grad_var < 50.0: s=0.2; score+=s; reasons.append(f"Low edge gradient variance"); score_components.append(s)
    if sat_entropy < 2.0: s=0.2; score+=s; reasons.append(f"Low color palette entropy"); score_components.append(s)
    if block_var > 0.6: s=0.3; score+=s; reasons.append(f"High ELA block variance - splicing"); radar["ELA Var"]=0.8; score_components.append(s)
    if ghost_var > 0.5 and not api_mode: s=0.25; score+=s; reasons.append(f"High JPEG Ghost variance"); score_components.append(s)
    if ghost_q40_var > 1.0 and not api_mode: s=0.3; score+=s; reasons.append(f"High Q40/Q90 JPEG Ghost variance - multi-generation splicing"); radar["Ghost Q40"]=0.85; score_components.append(s)
    if has_face and face_ela_score < 2.0: s=0.35; score+=s; reasons.append(f"Face region ELA suspiciously low"); score_components.append(s)
    if eye_asymmetry > 0.02: s=0.25; score+=s; reasons.append(f"Eye specular highlight asymmetry"); score_components.append(s)
    elif ela_mean < 1.5: s=0.15; score+=s; reasons.append("Low global ELA response - synthetic pipeline"); score_components.append(s)
    if ca_var < 10.0 and ca_var > 0.0: s=0.25; score+=s; reasons.append(f"Low Chromatic Aberration variance"); score_components.append(s)
    if light_asymmetry > 20.0: s=0.2; score+=s; reasons.append(f"High facial lighting asymmetry"); score_components.append(s)
    if eye_reflection_dist > 0.15: s=0.3; score+=s; reasons.append(f"Mismatched eye reflection centroids"); score_components.append(s)
    if dof_var < 100.0 and dof_var > 0.0: s=0.25; score+=s; reasons.append(f"Low depth-of-field gradient variance"); score_components.append(s)
    if vignette_ratio > 0.9: s=0.2; score+=s; reasons.append(f"Low lens vignetting gradient"); score_components.append(s)
    if nyquist_energy < 0.1: s=0.25; score+=s; reasons.append(f"Low CFA Nyquist energy - missing demosaicing artifacts"); score_components.append(s)

    score = float(min(0.95, score + 0.1))
    
    viz = None
    if HAS_MPL and not api_mode:
        fig = Figure(figsize=(8, 6), facecolor="#0d1117" if QApplication.instance().property("Theme") == "dark" else "#ffffff")
        ax1 = fig.add_subplot(231); ax1.imshow(np.log(mag + 1), cmap="inferno"); ax1.set_title("Global FFT", color="#c9d1d9" if QApplication.instance().property("Theme") == "dark" else "#24292f", fontsize=8); ax1.axis("off")
        ax2 = fig.add_subplot(232); ax2.imshow(ela_diff.clip(0, 30) / 30, cmap="hot")
        if has_face:
            for (x, y, fw, fh) in faces: ax2.add_patch(plt.Rectangle((x, y), fw, fh, fill=False, edgecolor='cyan', linewidth=2))
        ax2.set_title("ELA Diff", color="#c9d1d9" if QApplication.instance().property("Theme") == "dark" else "#24292f", fontsize=8); ax2.axis("off")
        ax3 = fig.add_subplot(233, polar=True)
        labels, stats = list(radar.keys()), list(radar.values())
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        stats += stats[:1]; angles += angles[:1]
        ax3.plot(angles, stats, color="#58a6ff", linewidth=2); ax3.fill(angles, stats, color="#58a6ff", alpha=0.3)
        ax3.set_xticks(angles[:-1]); ax3.set_xticklabels(labels, color="#c9d1d9" if QApplication.instance().property("Theme") == "dark" else "#24292f", fontsize=7)
        ax3.set_title("Anomaly Radar", color="#c9d1d9" if QApplication.instance().property("Theme") == "dark" else "#24292f", fontsize=8, pad=15); ax3.grid(color="#30363d")
        ax4 = fig.add_subplot(234); hm_img = Image.fromarray(np.uint8(ela_heatmap * 255 / (ela_heatmap.max() + 1e-9)))
        hm_img = hm_img.resize((w//4, h//4), Image.NEAREST); ax4.imshow(hm_img, cmap="magma")
        ax4.set_title("ELA Heatmap", color="#c9d1d9" if QApplication.instance().property("Theme") == "dark" else "#24292f", fontsize=8); ax4.axis("off")
        fig.tight_layout(); viz = fig

    ela_heatmap_resized = cv2.resize(ela_heatmap, (W, H)) if HAS_CV2 else ela_heatmap
    
    return {
        "score": score, "score_components": score_components, "radar": radar,
        "verdict_fake": "AI-Generated / Manipulated", "verdict_real": "Likely Authentic",
        "metrics": {
            "Resolution": f"{W}x{H}", "Global HF Ratio": f"{hf_ratio*100:.3f}%",
            "Noise Kurtosis": f"{noise_kurtosis:.2f}" if HAS_SCIPY else "N/A",
            "PRNU Cross-Corr": f"{prnu_corr:.2f}" if HAS_SCIPY else "N/A",
            "Ghost Q40 Var": f"{ghost_q40_var:.2f}" if not api_mode else "N/A"
        },
        "reasons": reasons, "viz": viz, "thumbnail": img.copy(),
        "ela_heatmap": ela_heatmap_resized, "faces": faces.tolist() if has_face else [],
        "methodology": "Inspired by DIRE (CVPR 2024). Adds Secondary JPEG Ghost Map (Q40 vs Q90). True images degrade uniformly; spliced regions often leave distinct ghost artifacts at high compression levels."
    }


# ===========================================================================
#  VIDEO DETECTION (Upgraded with Background Temporal Flicker)
# ===========================================================================
def detect_video(path, progress_cb, deep_scan, api_mode=False):
    if not HAS_CV2: return {"error": "opencv-python required for video analysis."}
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): return {"error": "Cannot open video file."}

    fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sample_n = min(30 if deep_scan else 15, max(5, total // 30 if total > 0 else 10))
    step = max(1, total // sample_n) if total > 0 else 1

    frames, indices, idx = [], [], 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % step == 0 and len(frames) < sample_n:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); indices.append(idx)
        idx += 1
    cap.release()

    per_frame_scores, flow_mags, mouth_variances, eye_variances, face_centers, fg_bg_flow_ratios, face_flicker_mses = [], [], [], [], [], [], []
    green_channel_means, cheek_textures, pupil_dark_ratios, eye_micro_saccades, skin_lbp_vars, teeth_brights, bg_flickers = [], [], [], [], [], [], []
    prev_gray, prev_face_patch = None, None
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    mouth_roi, eye_roi, face_box, forehead_roi, cheek_roi = None, None, None, None, None
    
    if len(frames) > 0:
        gray_first = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray_first, 1.3, 5)
        if len(faces) > 0:
            x, y, fw, fh = faces[0]; face_box = (x, y, fw, fh)
            mouth_roi = (x, int(y + fh*0.65), fw, int(fh*0.35))
            eye_roi = (x, int(y + fh*0.15), fw, int(fh*0.25))
            forehead_roi = (int(x + fw*0.3), int(y + fh*0.1), int(fw*0.4), int(fh*0.2))
            cheek_roi = (int(x + fw*0.1), int(y + fh*0.5), int(fw*0.2), int(fh*0.2))

    for i, f in enumerate(frames):
        gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32)
        F = np.fft.fftshift(np.fft.fft2(gray)); mag = np.abs(F)
        cy, cx = np.array(gray.shape) // 2
        Y, X = np.ogrid[:gray.shape[0], :gray.shape[1]]
        r = np.sqrt((Y - cy)**2 + (X - cx)**2).astype(int); rmax = min(gray.shape) // 2
        per_frame_scores.append(np.sum(mag[r > rmax // 4]**2) / (np.sum(mag**2) + 1e-9))

        if prev_gray is not None and gray.shape == prev_gray.shape:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag_flow = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flow_mags.append(float(mag_flow.mean()))
            if face_box:
                fx, fy, fw, fh = face_box
                fg_flow = mag_flow[fy:fy+fh, fx:fx+fw].mean()
                bg_mask = np.ones_like(mag_flow, dtype=bool); bg_mask[fy:fy+fh, fx:fx+fw] = False
                fg_bg_flow_ratios.append(float(fg_flow / (mag_flow[bg_mask].mean() + 1e-9)))
            
            if eye_roi:
                ex, ey, ew, eh = eye_roi
                eye_flow_mag = mag_flow[ey:ey+eh, ex:ex+ew]
                eye_micro_saccades.append(float(np.var(eye_flow_mag)))
        prev_gray = gray
        
        # NEW: Background Temporal Flicker
        # Track variance of a background region (top-left corner)
        if face_box:
            bg_patch = gray[0:H//4, 0:W//4]
            bg_flickers.append(float(np.std(bg_patch)))
        
        if mouth_roi: 
            mouth_patch = gray[mouth_roi[1]:mouth_roi[1]+mouth_roi[3], mouth_roi[0]:mouth_roi[0]+mouth_roi[2]]
            mouth_variances.append(float(np.std(mouth_patch)))
            teeth_brights.append(float(np.mean(mouth_patch > 180)))
            
        if eye_roi:
            eye_patch = gray[eye_roi[1]:eye_roi[1]+eye_roi[3], eye_roi[0]:eye_roi[0]+eye_roi[2]]
            eye_variances.append(float(np.std(eye_patch)))
            pupil_dark_ratios.append(float(np.mean(eye_patch < 50)))
        if forehead_roi:
            fx, fy, fw, fh = forehead_roi
            green_channel_means.append(float(np.mean(f[fy:fy+fh, fx:fx+fw, 1])))
        if cheek_roi:
            cx, cy, cw, ch = cheek_roi
            cheek_patch = gray[cy:cy+ch, cx:cx+cw]
            cheek_textures.append(float(cv2.Laplacian(cheek_patch, cv2.CV_64F).var()))
            skin_lbp_vars.append(fast_lbp_variance(cheek_patch.astype(np.uint8)))

        gray_u8 = gray.astype(np.uint8)
        curr_faces = face_cascade.detectMultiScale(gray_u8, 1.3, 5)
        if len(curr_faces) > 0:
            fx, fy, fw, fh = curr_faces[0]; face_centers.append((fx + fw/2, fy + fh/2))
            curr_face_patch = gray[fy:fy+fh, fx:fx+fw]
            if prev_face_patch is not None and curr_face_patch.shape == prev_face_patch.shape:
                face_flicker_mses.append(float(np.mean((curr_face_patch - prev_face_patch)**2)))
            prev_face_patch = curr_face_patch
            
        progress_cb.emit(40 + int(40 * i / max(1, len(frames))), f"Frame {i+1}/{len(frames)}")

    mean_hf = float(np.mean(per_frame_scores))
    flow_mean = float(np.mean(flow_mags)) if flow_mags else 0.0
    flow_smoothness = 1.0 - min(1.0, float(np.std(flow_mags) if flow_mags else 0.0) / (flow_mean + 1e-9))
    
    mouth_motion_ratio = float(np.std(mouth_variances) / (float(np.mean(mouth_variances)) + 1e-9)) if mouth_variances else 0.0
    eye_motion_ratio = float(np.std(eye_variances) / (float(np.mean(eye_variances)) + 1e-9)) if eye_variances else 0.0
    face_jitter = float(np.mean(np.std(face_centers, axis=0))) if len(face_centers) > 2 else 0.0
    
    pose_radius = 0.0
    if len(face_centers) > 5:
        xs = [p[0] for p in face_centers]; ys = [p[1] for p in face_centers]
        pose_radius = float(np.std(xs) + np.std(ys))
    
    fg_bg_mean_ratio = float(np.mean(fg_bg_flow_ratios)) if fg_bg_flow_ratios else 1.0
    flicker_std = float(np.std(face_flicker_mses)) if face_flicker_mses else 0.0
    av_sync_corr = 0.0
    if len(mouth_variances) > 4 and len(flow_mags) > 4:
        mv = np.array(mouth_variances[:len(flow_mags)]); fm = np.array(flow_mags)
        mv = (mv - np.mean(mv)) / (np.std(mv) + 1e-9); fm = (fm - np.mean(fm)) / (np.std(fm) + 1e-9)
        av_sync_corr = float(np.max(np.abs(np.correlate(mv, fm, 'full'))) / (len(mv) * 1.0))
    rppg_var = float(np.std(green_channel_means)) if green_channel_means else 0.0
    cheek_texture_std = float(np.std(cheek_textures)) if cheek_textures else 0.0
    pupil_var = float(np.std(pupil_dark_ratios)) if pupil_dark_ratios else 0.0
    saccade_var = float(np.std(eye_micro_saccades)) if eye_micro_saccades else 0.0
    lbp_var = float(np.std(skin_lbp_vars)) if skin_lbp_vars else 0.0
    teeth_var = float(np.std(teeth_brights)) if teeth_brights else 0.0
    bg_flicker_std = float(np.std(bg_flickers)) if bg_flickers else 0.0
    
    hrv_var = 0.0
    if green_channel_means and HAS_SCIPY:
        signal = np.array(green_channel_means)
        if len(signal) > 10:
            smoothed = np.convolve(signal, np.ones(5)/5, mode='valid')
            peaks = scipy_signal.find_peaks(smoothed)[0]
            if len(peaks) > 2:
                pp_intervals = np.diff(peaks)
                hrv_var = float(np.var(pp_intervals))

    score, reasons, radar, score_components = 0.0, [], {"HF Energy": 0, "Flow": 0, "Lip Sync": 0, "rPPG": 0, "BG Flicker": 0}, []
    if mean_hf < 0.02: s=0.15; score+=s; reasons.append(f"Low mean HF energy"); radar["HF Energy"]=0.8; score_components.append(s)
    if flow_smoothness > 0.7 and flow_mean > 0.1: s=0.3; score+=s; reasons.append(f"Optical flow too smooth"); radar["Flow"]=0.9; score_components.append(s)
    if mouth_roi and mouth_motion_ratio < 0.1 and flow_mean > 0.1: s=0.35; score+=s; reasons.append(f"Lip-sync anomaly"); radar["Lip Sync"]=0.95; score_components.append(s)
    if eye_roi and eye_motion_ratio < 0.05 and flow_mean > 0.1: s=0.3; score+=s; reasons.append(f"Eye region unnaturally static"); score_components.append(s)
    if (fg_bg_mean_ratio > 3.0 or fg_bg_mean_ratio < 0.33) and flow_mean > 0.1: s=0.25; score+=s; reasons.append(f"FG/BG flow detachment"); score_components.append(s)
    if face_jitter > 0 and (face_jitter < 0.5 or face_jitter > 15.0): s=0.25; score+=s; reasons.append(f"Unnatural face jitter"); score_components.append(s)
    if flicker_std > 50.0: s=0.3; score+=s; reasons.append(f"High face flicker MSE"); score_components.append(s)
    if av_sync_corr < 0.2 and flow_mean > 0.1: s=0.25; score+=s; reasons.append(f"Low AV sync correlation"); score_components.append(s)
    if rppg_var < 0.5 and flow_mean > 0.1: s=0.35; score+=s; reasons.append(f"Absence of rPPG pulse signal"); radar["rPPG"]=0.95; score_components.append(s)
    if cheek_texture_std < 5.0 and len(cheek_textures) > 3: s=0.2; score+=s; reasons.append(f"Low cheek texture variance"); score_components.append(s)
    if pupil_var < 0.001 and len(pupil_dark_ratios) > 3: s=0.3; score+=s; reasons.append(f"Static pupil light response"); score_components.append(s)
    if pose_radius < 5.0 and len(face_centers) > 5: s=0.25; score+=s; reasons.append(f"Unnatural head pose radius"); score_components.append(s)
    if saccade_var < 0.001 and len(eye_micro_saccades) > 3: s=0.3; score+=s; reasons.append(f"Absence of micro-saccade eye jitter"); score_components.append(s)
    if lbp_var < 0.0001 and len(skin_lbp_vars) > 3: s=0.25; score+=s; reasons.append(f"Low skin LBP variance - plastic deepfake texture"); score_components.append(s)
    if hrv_var < 0.5 and hrv_var > 0.0 and rppg_var > 0.5: s=0.25; score+=s; reasons.append(f"Low Heart Rate Variability"); score_components.append(s)
    if teeth_var < 0.001 and len(teeth_brights) > 3 and flow_mean > 0.1: s=0.2; score+=s; reasons.append(f"Low teeth brightness variance - static mouth render"); score_components.append(s)
    
    # Deepfakes introduce global noise that flickers unnaturally
    if bg_flicker_std > 10.0 and len(bg_flickers) > 3:
        s=0.2; score+=s; reasons.append(f"High background temporal flicker ({bg_flicker_std:.2f}) - rendering noise"); radar["BG Flicker"]=0.85; score_components.append(s)

    score = float(min(0.95, score + 0.1))
    
    viz = None
    if HAS_MPL and not api_mode:
        fig = Figure(figsize=(8, 4), facecolor="#0d1117" if QApplication.instance().property("Theme") == "dark" else "#ffffff")
        ax1 = fig.add_subplot(141); ax1.plot(indices[:len(per_frame_scores)], per_frame_scores, marker="o", color="#58a6ff")
        ax1.set_title("Per-frame HF Energy", color="#c9d1d9" if QApplication.instance().property("Theme") == "dark" else "#24292f", fontsize=9)
        for spine in ax1.spines.values(): spine.set_color("#30363d" if QApplication.instance().property("Theme") == "dark" else "#d0d7de")
        ax2 = fig.add_subplot(142); ax2.plot(flow_mags, color="#f85149", marker="s")
        ax2.set_title("Optical Flow", color="#c9d1d9" if QApplication.instance().property("Theme") == "dark" else "#24292f", fontsize=9)
        for spine in ax2.spines.values(): spine.set_color("#30363d" if QApplication.instance().property("Theme") == "dark" else "#d0d7de")
        ax3 = fig.add_subplot(143)
        if mouth_variances: ax3.plot(mouth_variances, color="#3fb950", marker="^", label="Mouth")
        if green_channel_means: ax3.plot(green_channel_means, color="#58a6ff", marker="o", label="rPPG")
        ax3.set_title("Biological & Motion", color="#c9d1d9" if QApplication.instance().property("Theme") == "dark" else "#24292f", fontsize=9); ax3.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=7)
        for spine in ax3.spines.values(): spine.set_color("#30363d" if QApplication.instance().property("Theme") == "dark" else "#d0d7de")
        ax4 = fig.add_subplot(144, polar=True)
        labels, stats = list(radar.keys()), list(radar.values())
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        stats += stats[:1]; angles += angles[:1]
        ax4.plot(angles, stats, color="#58a6ff", linewidth=2); ax4.fill(angles, stats, color="#58a6ff", alpha=0.3)
        ax4.set_xticks(angles[:-1]); ax4.set_xticklabels(labels, color="#c9d1d9" if QApplication.instance().property("Theme") == "dark" else "#24292f", fontsize=8)
        ax4.set_title("Anomaly Radar", color="#c9d1d9" if QApplication.instance().property("Theme") == "dark" else "#24292f", fontsize=9, pad=15); ax4.grid(color="#30363d")
        fig.tight_layout(); viz = fig

    return {
        "score": score, "score_components": score_components, "radar": radar,
        "verdict_fake": "AI-Generated / Deepfake", "verdict_real": "Likely Authentic",
        "metrics": {
            "Resolution": f"{W}x{H}", "FPS": f"{fps:.2f}", "Flow Smoothness": f"{flow_smoothness:.3f}",
            "Face Flicker Std": f"{flicker_std:.2f}" if face_flicker_mses else "N/A",
            "rPPG Variance": f"{rppg_var:.2f}" if green_channel_means else "N/A",
            "BG Flicker Std": f"{bg_flicker_std:.2f}" if bg_flickers else "N/A"
        },
        "reasons": reasons, "viz": viz, "thumbnails": frames[:6],
        "methodology": "Inspired by LipForensics (CVPR 2021). Adds Background Temporal Flicker tracking. Deepfakes often introduce global high-frequency noise that flickers at a different rate than the face."
    }


# ===========================================================================
#  TEXT DETECTION (Upgraded with Pronoun Density Skew)
# ===========================================================================
def detect_text(path, progress_cb, deep_scan, custom_lexicon_str="", custom_rules_str="", api_mode=False):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f: text = f.read()
    else: text = path
    if len(text.strip()) < 20: return {"error": "Text too short (min 20 chars)."}

    active_lexicon = set(DEFAULT_LLM_LEXICON)
    if custom_lexicon_str:
        active_lexicon.update([w.strip().lower() for w in custom_lexicon_str.split(",") if w.strip()])

    words = text.lower().split()
    words = [w.strip(".,!?;:\"'()[]{}") for w in words if w.strip(".,!?;:\"'()[]{}")]
    n_words = len(words)
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    n_sent = max(1, len(sentences))

    ttr = len(set(words)) / max(1, n_words)
    mtld = calculate_mtld(words)
    freqs = Counter(words)
    sent_lens = [len(s.split()) for s in sentences]
    mean_len = float(np.mean(sent_lens)) if sent_lens else 0
    burstiness = float(np.std(sent_lens)) / (mean_len + 1e-9) if sent_lens else 0

    llm_word_rate = sum(freqs.get(w, 0) for w in active_lexicon) / max(1, n_words)
    transition_rate = sum(freqs.get(w, 0) for w in LLM_TRANSITIONS) / max(1, n_sent)

    connective_count = sum(freqs.get(w, 0) for w in CAUSAL_CONNECTIVES)
    connective_density = connective_count / max(1, n_words) * 100

    total_syllables = sum(count_syllables(w) for w in words)
    flesch_ease = 206.835 - 1.015 * (n_words / n_sent) - 84.6 * (total_syllables / n_words) if n_words > 0 and n_sent > 0 else 0

    sentences_raw = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    missed_caps = sum(1 for s in sentences_raw if s and not s[0].isupper()) / max(1, len(sentences_raw))
    em_dash_rate = (text.count('—') + text.count('–')) / max(1, n_words)

    three_grams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
    ngram_repeats = sum(1 for c in Counter(three_grams).values() if c > 1)
    ngram_repeat_rate = ngram_repeats / max(1, len(three_grams))

    starter_entropy = 0.0
    if n_sent > 3:
        starters = [s.split()[0].lower() if s.split() else "" for s in sentences_raw]
        starter_counts = Counter(starters)
        probs = [c / len(starters) for c in starter_counts.values()]
        starter_entropy = float(scipy_entropy(probs)) if HAS_SCIPY else 0.0

    punct_entropy = 0.0
    if HAS_SCIPY:
        punct_counts = Counter(c for c in text if c in ".,;:!?-—\"'()[]")
        if sum(punct_counts.values()) > 0:
            probs = [c / sum(punct_counts.values()) for c in punct_counts.values()]
            punct_entropy = float(scipy_entropy(probs))

    markdown_pattern = r'(\*\*.*?\*\*|#+\s|```|__.*?__|`[^`]+`)'
    markdown_matches = re.findall(markdown_pattern, text)
    markdown_density = len(markdown_matches) / max(1, n_words) * 100

    content_words = [w for w in words if w not in FUNCTION_WORDS]
    lexical_density = len(content_words) / max(1, n_words) * 100

    typo_space = len(re.findall(r'[.,;:!?][a-zA-Z]', text))
    typo_rate = typo_space / max(1, n_words)

    fw_skew = 0.0
    if HAS_SCIPY:
        fw_counts = [freqs.get(w, 0) for w in FUNCTION_WORDS if freqs.get(w, 0) > 0]
        if len(fw_counts) > 5:
            fw_skew = float(scipy_skew(fw_counts))

    double_quotes = text.count('"')
    single_quotes = text.count("'")
    quote_mismatch = (double_quotes % 2 != 0) or (single_quotes % 2 != 0)

    # NEW: Pronoun Density Skew
    pronoun_count = sum(freqs.get(w, 0) for w in PRONOUNS)
    pronoun_density = pronoun_count / max(1, n_words) * 100

    semantic_var = 0.0
    semantic_curvature = 0.0
    if n_sent > 2:
        all_words = list(set(words))
        sent_vectors = []
        for s in sentences:
            w_counts = Counter(s.lower().split())
            sent_vectors.append(np.array([w_counts.get(w, 0) for w in all_words], dtype=np.float32))
        def cosine_sim(v1, v2): return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        sims = [cosine_sim(sent_vectors[i], sent_vectors[i+1]) for i in range(len(sent_vectors)-1)]
        semantic_var = float(np.std(sims)) if sims else 0.0
        if len(sims) > 2:
            semantic_curvature = float(np.std(np.diff(sims)))

    zipf_slope = -1.0
    if n_words > 50:
        sorted_freqs = np.array(sorted(freqs.values(), reverse=True))
        zipf_slope, _ = np.polyfit(np.log(np.arange(1, len(sorted_freqs) + 1)), np.log(sorted_freqs), 1)

    score, reasons, radar, score_components = 0.0, [], {"Burstiness": 0, "Lexicon": 0, "MTLD": 0, "Zipf": 0, "Pronouns": 0}, []
    if burstiness < 0.45: s=0.25; score+=s; reasons.append(f"Low sentence-length burstiness"); radar["Burstiness"]=0.9; score_components.append(s)
    if llm_word_rate > 0.005: s=0.25; score+=s; reasons.append(f"High LLM lexical fingerprint"); radar["Lexicon"]=0.95; score_components.append(s)
    if semantic_var < 0.15 and n_sent > 3: s=0.3; score+=s; reasons.append(f"Low semantic jump variance"); score_components.append(s)
    if missed_caps < 0.05 and len(sentences_raw) > 5: s=0.15; score+=s; reasons.append("Perfect sentence capitalization"); score_components.append(s)
    if em_dash_rate > 0.005: s=0.25; score+=s; reasons.append(f"High em-dash usage"); score_components.append(s)
    if zipf_slope > -0.80 or zipf_slope < -1.20: s=0.25; score+=s; reasons.append(f"Zipf's Law deviation"); score_components.append(s)
    if transition_rate > 0.2: s=0.2; score+=s; reasons.append(f"High transition word overuse"); score_components.append(s)
    if ngram_repeat_rate > 0.05: s=0.25; score+=s; reasons.append(f"High 3-gram structural repetition"); score_components.append(s)
    if starter_entropy < 1.5 and n_sent > 3: s=0.25; score+=s; reasons.append(f"Low sentence starter entropy"); score_components.append(s)
    if mtld < 30.0 and n_words > 50: s=0.3; score+=s; reasons.append(f"Low MTLD lexical diversity"); radar["MTLD"]=0.9; score_components.append(s)
    if punct_entropy < 1.0 and n_words > 50: s=0.2; score+=s; reasons.append(f"Low punctuation entropy"); score_components.append(s)
    if markdown_density > 2.0 and n_words > 20: s=0.25; score+=s; reasons.append(f"High Markdown formatting density"); score_components.append(s)
    if lexical_density > 55.0 and n_words > 50: s=0.25; score+=s; reasons.append(f"High lexical density"); score_components.append(s)
    if typo_rate == 0.0 and n_words > 50: s=0.25; score+=s; reasons.append(f"Zero punctuation spacing typos"); score_components.append(s)
    if semantic_curvature < 0.1 and n_sent > 4: s=0.25; score+=s; reasons.append(f"Low semantic curvature"); score_components.append(s)
    if fw_skew < 1.5 and n_words > 50: s=0.2; score+=s; reasons.append(f"Low function word skewness"); score_components.append(s)
    if connective_density > 1.5 and n_words > 20: s=0.2; score+=s; reasons.append(f"High causal/contrastive connective density"); score_components.append(s)
    if not quote_mismatch and n_words > 50: s=0.15; score+=s; reasons.append(f"Perfectly matched quotation marks - LLM artifact"); score_components.append(s)
    
    # LLMs overuse pronouns due to RLHF inclusivity tuning
    if pronoun_density > 12.0 and n_words > 50:
        s=0.2; score+=s; reasons.append(f"High pronoun density ({pronoun_density:.1f}%) - RLHF inclusivity artifact"); radar["Pronouns"]=0.85; score_components.append(s)

    rules = parse_custom_rules(custom_rules_str)
    if rules:
        mod, matched = apply_custom_rules(text, rules)
        if mod != 0.0:
            score += mod
            reasons.extend(matched)
            score_components.append(mod)

    score = float(max(0.0, min(0.95, score + 0.1)))
    
    viz = None
    if HAS_MPL and not api_mode:
        fig = Figure(figsize=(8, 4), facecolor="#0d1117" if QApplication.instance().property("Theme") == "dark" else "#ffffff")
        ax1 = fig.add_subplot(131); ax1.hist(sent_lens, bins=min(15, max(5, len(sent_lens) // 2)), color="#58a6ff", edgecolor="#0d1117")
        ax1.axvline(mean_len, color="#f85149", linestyle="--", label=f"μ={mean_len:.1f}")
        ax1.set_title("Sentence Length", color="#c9d1d9" if QApplication.instance().property("Theme") == "dark" else "#24292f", fontsize=9)
        for spine in ax1.spines.values(): spine.set_color("#30363d" if QApplication.instance().property("Theme") == "dark" else "#d0d7de")
        ax2 = fig.add_subplot(132)
        if n_words > 50:
            ax2.loglog(np.arange(1, len(freqs)+1), sorted(freqs.values(), reverse=True), color="#3fb950", marker=".", linestyle="none")
            ax2.set_title(f"Zipf Dist (Slope={zipf_slope:.2f})", color="#c9d1d9" if QApplication.instance().property("Theme") == "dark" else "#24292f", fontsize=9)
        for spine in ax2.spines.values(): spine.set_color("#30363d" if QApplication.instance().property("Theme") == "dark" else "#d0d7de")
        ax3 = fig.add_subplot(133, polar=True)
        labels, stats = list(radar.keys()), list(radar.values())
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        stats += stats[:1]; angles += angles[:1]
        ax3.plot(angles, stats, color="#58a6ff", linewidth=2); ax3.fill(angles, stats, color="#58a6ff", alpha=0.3)
        ax3.set_xticks(angles[:-1]); ax3.set_xticklabels(labels, color="#c9d1d9" if QApplication.instance().property("Theme") == "dark" else "#24292f", fontsize=8)
        ax3.set_title("Anomaly Radar", color="#c9d1d9" if QApplication.instance().property("Theme") == "dark" else "#24292f", fontsize=9, pad=15); ax3.grid(color="#30363d")
        fig.tight_layout(); viz = fig

    return {
        "score": score, "score_components": score_components, "radar": radar,
        "verdict_fake": "AI-Generated Text", "verdict_real": "Likely Human-Written",
        "metrics": {
            "Total Words": str(n_words), "Type-Token Ratio": f"{ttr:.4f}",
            "MTLD Lexical Div": f"{mtld:.1f}", "Zipf Slope": f"{zipf_slope:.3f}" if n_words > 50 else "N/A",
            "Lexical Density": f"{lexical_density:.1f}%",
            "Pronoun Density": f"{pronoun_density:.1f}%"
        },
        "reasons": reasons, "viz": viz,
        "methodology": "Inspired by DetectGPT (ICML 2023). Adds Pronoun Density Skew. LLMs are heavily RLHF-tuned for inclusivity, often resulting in an unnaturally high usage of gender-neutral pronouns."
    }


# ===========================================================================
#  REST API SERVER
# ===========================================================================
class APIServerThread(QThread):
    status = Signal(str)
    def __init__(self, port=8765):
        super().__init__(); self.port = port; self.running = True

    def run(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format, *args): pass
            def do_POST(self):
                if self.path == "/detect":
                    try:
                        length = int(self.headers['Content-Length'])
                        post_data = self.rfile.read(length)
                        payload = json.loads(post_data.decode('utf-8'))
                        
                        modality = payload.get("modality")
                        b64_data = payload.get("data")
                        if not modality or not b64_data:
                            self.send_response(400); self.end_headers(); self.wfile.write(b'{"error":"Missing modality or data"}'); return
                        
                        file_data = base64.b64decode(b64_data)
                        ext = ".wav" if modality == "audio" else ".png" if modality == "image" else ".mp4" if modality == "video" else ".txt"
                        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                            tmp.write(file_data); tmp_path = tmp.name
                        
                        if modality == "audio": res = detect_audio(tmp_path, lambda p, m: None, False, api_mode=True)
                        elif modality == "image": res = detect_image(tmp_path, lambda p, m: None, False, api_mode=True)
                        elif modality == "video": res = detect_video(tmp_path, lambda p, m: None, False, api_mode=True)
                        elif modality == "text": res = detect_text(tmp_path, lambda p, m: None, False, "", "", api_mode=True)
                        else: res = {"error": "Invalid modality"}
                        
                        os.remove(tmp_path)
                        
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps(res).encode('utf-8'))
                    except Exception as e:
                        self.send_response(500); self.end_headers(); self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
                else:
                    self.send_response(404); self.end_headers()
        
        httpd = socketserver.TCPServer(("localhost", self.port), Handler)
        self.status.emit(f"API Server running on http://localhost:{self.port}")
        while self.running:
            httpd.handle_request()
        httpd.server_close()

    def stop(self):
        self.running = False
        import urllib.request
        try: urllib.request.urlopen(f"http://localhost:{self.port}/shutdown")
        except: pass


# ===========================================================================
#  LIVE WEBCAM DETECTION TAB
# ===========================================================================
class WebcamTab(QWidget):
    def __init__(self):
        super().__init__(); self.timer = QTimer(); self.timer.timeout.connect(self.update_frame)
        self.cap = None; self.face_cascade = None
        if HAS_CV2: self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.prev_gray = None; self.prev_face_patch = None
        self.liveness_scores = []
        self.green_history = deque(maxlen=150)
        self.time_history = deque(maxlen=150)
        self.start_time = 0.0
        self.build_ui()

    def build_ui(self):
        outer = QVBoxLayout(self); outer.setSpacing(12)
        outer.addWidget(QLabel("🎥  LIVE WEBCAM DETECTION", objectName="title"))
        outer.addWidget(QLabel("Real-time liveness analysis. Tracks face jitter, eye/blink variance, mouth motion, and heart rate (rPPG).", objectName="subtitle"))

        ctrl = QHBoxLayout()
        self.btn_start = QPushButton("▶  Start Camera"); self.btn_start.clicked.connect(self.start_cam)
        self.btn_stop = QPushButton("⏹  Stop Camera", objectName="danger"); self.btn_stop.clicked.connect(self.stop_cam)
        ctrl.addWidget(self.btn_start); ctrl.addWidget(self.btn_stop); ctrl.addStretch()
        outer.addLayout(ctrl)

        self.video_label = QLabel("Camera offline"); self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#0d1117; border:1px solid #21262d; border-radius:6px; color:#8b949e;")
        self.video_label.setMinimumHeight(480)
        outer.addWidget(self.video_label, 1)

        self.lbl_stats = QLabel("Liveness Score: N/A  |  HR: N/A bpm"); self.lbl_stats.setStyleSheet("color:#58a6ff; font-size:18px; font-weight:700; padding:10px;")
        outer.addWidget(self.lbl_stats)

    def start_cam(self):
        if not HAS_CV2: return QMessageBox.warning(self, "Error", "OpenCV required for webcam.")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened(): return QMessageBox.warning(self, "Error", "Could not open webcam.")
        import time; self.start_time = time.time()
        self.green_history.clear(); self.time_history.clear()
        self.timer.start(33)

    def stop_cam(self):
        self.timer.stop()
        if self.cap: self.cap.release(); self.cap = None
        self.video_label.setText("Camera offline"); self.video_label.setPixmap(QPixmap())
        self.lbl_stats.setText("Liveness Score: N/A  |  HR: N/A bpm")
        self.liveness_scores.clear()

    def update_frame(self):
        import time
        ret, frame = self.cap.read()
        if not ret: return
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w, _ = frame_rgb.shape
        color = (0, 255, 0) if not self.prev_gray else (255, 255, 255)
        
        if self.face_cascade:
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) > 0:
                fx, fy, fw, fh = faces[0]
                cv2.rectangle(frame_rgb, (fx, fy), (fx+fw, fy+fh), color, 2)
                eye_y = int(fy + fh*0.15); eye_h = int(fh*0.25)
                mouth_y = int(fy + fh*0.65); mouth_h = int(fh*0.35)
                forehead_y = int(fy + fh*0.1); forehead_h = int(fh*0.2)
                
                cv2.rectangle(frame_rgb, (fx, eye_y), (fx+fw, eye_y+eye_h), (255, 0, 0), 1)
                cv2.rectangle(frame_rgb, (fx, mouth_y), (fx+fw, mouth_y+mouth_h), (0, 0, 255), 1)
                cv2.rectangle(frame_rgb, (int(fx+fw*0.3), forehead_y), (int(fx+fw*0.7), forehead_y+forehead_h), (0, 255, 255), 1)
                
                face_patch = gray[fy:fy+fh, fx:fx+fw].astype(np.float32)
                eye_patch = gray[eye_y:eye_y+eye_h, fx:fx+fw].astype(np.float32)
                mouth_patch = gray[mouth_y:mouth_y+mouth_h, fx:fx+fw].astype(np.float32)
                forehead_patch_rgb = frame_rgb[forehead_y:forehead_y+forehead_h, int(fx+fw*0.3):int(fx+fw*0.7), :]
                
                flicker = 0.0
                if self.prev_face_patch is not None and face_patch.shape == self.prev_face_patch.shape:
                    flicker = float(np.mean((face_patch - self.prev_face_patch)**2))
                eye_var = float(np.std(eye_patch)); mouth_var = float(np.std(mouth_patch))
                self.liveness_scores.append({"flicker": flicker, "eye": eye_var, "mouth": mouth_var})
                if len(self.liveness_scores) > 30: self.liveness_scores.pop(0)
                self.prev_face_patch = face_patch
                
                green_mean = float(np.mean(forehead_patch_rgb[:, :, 1]))
                self.green_history.append(green_mean)
                self.time_history.append(time.time() - self.start_time)
                
                avg_flicker = np.mean([s["flicker"] for s in self.liveness_scores]) if self.liveness_scores else 0
                avg_eye = np.mean([s["eye"] for s in self.liveness_scores]) if self.liveness_scores else 0
                avg_mouth = np.mean([s["mouth"] for s in self.liveness_scores]) if self.liveness_scores else 0
                
                score = 100.0
                if avg_flicker > 50.0 or avg_flicker < 1.0: score -= 30
                if avg_eye < 10.0: score -= 30
                if avg_mouth < 10.0: score -= 30
                
                hr_text = "Calculating..."
                if len(self.green_history) >= 60:
                    signal = np.array(self.green_history) - np.mean(self.green_history)
                    if np.max(np.abs(signal)) > 1e-3:
                        fs = 30.0
                        spectrum = np.abs(np.fft.fft(signal))
                        freqs = np.fft.fftfreq(len(signal), 1/fs)
                        valid = (freqs > 0.7) & (freqs < 3.0)
                        if np.any(valid):
                            peak_freq = freqs[np.argmax(spectrum[valid])]
                            hr = peak_freq * 60
                            hr_text = f"{int(hr)} bpm"
                            if hr < 40 or hr > 180:
                                score -= 20
                                hr_text = f"{int(hr)} bpm (Abnormal!)"
                        else: hr_text = "N/A"
                    else: hr_text = "N/A (Static)"
                
                self.lbl_stats.setText(f"Liveness Score: {max(0, int(score))}%  |  HR: {hr_text}")
                if score < 50: self.lbl_stats.setStyleSheet("color:#f85149; font-size:18px; font-weight:700; padding:10px;")
                else: self.lbl_stats.setStyleSheet("color:#3fb950; font-size:18px; font-weight:700; padding:10px;")
            else: self.lbl_stats.setText("Liveness Score: N/A  |  HR: N/A (No face)")
        
        self.prev_gray = gray
        qimg = QImage(frame_rgb.data, w, h, 3*w, QImage.Format_RGB888)
        pm = QPixmap.fromImage(qimg).scaledToWidth(640, Qt.SmoothTransformation)
        self.video_label.setPixmap(pm)

    def closeEvent(self, event):
        self.stop_cam(); event.accept()


# ===========================================================================
#  SETTINGS TAB
# ===========================================================================
class SettingsTab(QWidget):
    settings_changed = Signal(dict)
    def __init__(self, config):
        super().__init__(); self.config = config; self.build_ui()

    def build_ui(self):
        outer = QVBoxLayout(self); outer.setSpacing(12)
        outer.addWidget(QLabel("⚙️  GLOBAL SETTINGS", objectName="title"))
        outer.addWidget(QLabel("Adjust global sensitivity, UI theme, and settings. Changes are saved automatically.", objectName="subtitle"))

        ui_box = QGroupBox("User Interface")
        ui_layout = QHBoxLayout(ui_box)
        ui_layout.addWidget(QLabel("Theme:"))
        self.combo_theme = QComboBox(); self.combo_theme.addItems(["Dark", "Light"])
        self.combo_theme.setCurrentText(self.config.get("theme", "dark").capitalize())
        self.combo_theme.currentTextChanged.connect(self.on_change)
        ui_layout.addWidget(self.combo_theme); ui_layout.addStretch()
        outer.addWidget(ui_box)

        sens_box = QGroupBox("Modality Sensitivity (0=Strict, 100=Permissive)")
        sens_layout = QGridLayout(sens_box)
        
        self.sliders = {}
        modalities = ["audio", "image", "video", "text"]
        for i, mod in enumerate(modalities):
            sens_layout.addWidget(QLabel(f"{mod.capitalize()} Sensitivity:"), i, 0)
            slider = QSlider(Qt.Horizontal); slider.setRange(0, 100); slider.setValue(self.config.get(f"sensitivity_{mod}", 50))
            slider.valueChanged.connect(self.on_change)
            sens_layout.addWidget(slider, i, 1)
            lbl = QLabel(f"{slider.value()}%")
            lbl.setMinimumWidth(40)
            slider.valueChanged.connect(lambda v, l=lbl: l.setText(f"{v}%"))
            sens_layout.addWidget(lbl, i, 2)
            self.sliders[mod] = slider
            
        outer.addWidget(sens_box)

        lex_box = QGroupBox("Custom LLM Lexicon (Comma-Separated)")
        lex_layout = QVBoxLayout(lex_box)
        self.lexicon_input = QLineEdit(self.config.get("custom_lexicon", ""))
        self.lexicon_input.setPlaceholderText("e.g., underscore, delve, tapestry, custom_word")
        self.lexicon_input.textChanged.connect(self.on_change)
        lex_layout.addWidget(self.lexicon_input)
        outer.addWidget(lex_box)

        rules_box = QGroupBox("Custom Text Rules (Format: word=score, e.g., midjourney=20, as an ai=-50)")
        rules_layout = QVBoxLayout(rules_box)
        self.rules_input = QLineEdit(self.config.get("custom_rules", ""))
        self.rules_input.setPlaceholderText("word=score, word2=score2")
        self.rules_input.textChanged.connect(self.on_change)
        rules_layout.addWidget(self.rules_input)
        outer.addWidget(rules_box)

        api_box = QGroupBox("REST API Server")
        api_layout = QHBoxLayout(api_box)
        api_layout.addWidget(QLabel("Port:"))
        self.port_input = QSpinBox(); self.port_input.setRange(1024, 65535); self.port_input.setValue(self.config.get("api_port", 8765))
        self.port_input.valueChanged.connect(self.on_change)
        api_layout.addWidget(self.port_input); api_layout.addStretch()
        outer.addWidget(api_box)
        
        outer.addStretch()

    def on_change(self):
        self.config["theme"] = self.combo_theme.currentText().lower()
        self.config["sensitivity_audio"] = self.sliders["audio"].value()
        self.config["sensitivity_image"] = self.sliders["image"].value()
        self.config["sensitivity_video"] = self.sliders["video"].value()
        self.config["sensitivity_text"] = self.sliders["text"].value()
        self.config["custom_lexicon"] = self.lexicon_input.text()
        self.config["custom_rules"] = self.rules_input.text()
        self.config["api_port"] = self.port_input.value()
        save_config(self.config)
        self.settings_changed.emit(self.config)


# ===========================================================================
#  BASE DETECTION TAB (With XML Export & Global Hotkey)
# ===========================================================================
class URLDownloadWorker(QThread):
    finished = Signal(str)
    error = Signal(str)
    def __init__(self, url):
        super().__init__(); self.url = url
    def run(self):
        try:
            req = urllib.request.Request(self.url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = response.read()
            tmp_path = os.path.join(tempfile.gettempdir(), "deepguard_url_img.png")
            with open(tmp_path, "wb") as f: f.write(data)
            self.finished.emit(tmp_path)
        except Exception as e:
            self.error.emit(str(e))

class DetectionTab(QWidget):
    result_logged = Signal(dict)

    def __init__(self, modality, accept_label, file_filter, sensitivity=50):
        super().__init__()
        self.modality, self.worker, self.current_path, self.last_result = modality, None, None, None
        self.sensitivity, self.deep_scan = sensitivity, False
        self.base_pixmap = None
        self.url_worker = None
        self.custom_lexicon = ""
        self.custom_rules = ""
        self.build_ui(accept_label, file_filter)
        self.setAcceptDrops(True)
        
        if modality == "image":
            self.shortcut_paste = QShortcut(QKeySequence.Paste, self)
            self.shortcut_paste.activated.connect(self.paste_image)
            
        # Global Hotkey: Ctrl+Enter / Cmd+Return to run detection
        self.shortcut_run = QShortcut(QKeySequence("Ctrl+Return"), self)
        self.shortcut_run.activated.connect(self.run_detection)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls(): event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        path = event.mimeData().urls()[0].toLocalFile()
        if path:
            self.current_path = path
            self.lbl_path.setText(f"Loaded: {os.path.basename(path)}"); self.lbl_path.setStyleSheet("color:#3fb950;")
            self.btn_analyze.setEnabled(True); self.show_preview(path)

    def paste_image(self):
        cb = QApplication.clipboard()
        if cb.image().isNull(): return
        img = cb.image()
        tmp_path = os.path.join(tempfile.gettempdir(), "deepguard_clipboard.png")
        img.save(tmp_path, "PNG")
        self.current_path = tmp_path
        self.lbl_path.setText(f"Loaded: Clipboard Image"); self.lbl_path.setStyleSheet("color:#3fb950;")
        self.btn_analyze.setEnabled(True)
        self.show_preview(tmp_path)

    def build_ui(self, accept_label, file_filter):
        outer = QVBoxLayout(self); outer.setSpacing(12)
        outer.addWidget(QLabel(f"{self.modality.upper()} DETECTION", objectName="title"))
        outer.addWidget(QLabel(f"Load, Drag & Drop, Paste (Images), or URL (Images) to analyze.", objectName="subtitle"))

        ctrl = QHBoxLayout()
        self.btn_load = QPushButton(f"📁  Load {self.modality.capitalize()}")
        self.btn_load.clicked.connect(lambda: self.load_file(file_filter))
        self.btn_analyze = QPushButton("🔬  Run Detection", objectName="secondary")
        self.btn_analyze.setEnabled(False); self.btn_analyze.clicked.connect(self.run_detection)
        self.chk_deep = QCheckBox("Deep Scan"); self.chk_deep.setStyleSheet("color:#c9d1d9; font-weight:600;")
        
        self.btn_export_md = QPushButton("📄  Export MD", objectName="secondary")
        self.btn_export_md.setEnabled(False); self.btn_export_md.clicked.connect(lambda: self.export_report("md"))
        self.btn_export_pdf = QPushButton("📄  Export PDF", objectName="secondary")
        self.btn_export_pdf.setEnabled(False); self.btn_export_pdf.clicked.connect(lambda: self.export_report("pdf"))
        self.btn_export_html = QPushButton("📄  Export HTML", objectName="secondary")
        self.btn_export_html.setEnabled(False); self.btn_export_html.clicked.connect(lambda: self.export_report("html"))
        self.btn_export_csv = QPushButton("📄  Export CSV", objectName="secondary")
        self.btn_export_csv.setEnabled(False); self.btn_export_csv.clicked.connect(lambda: self.export_report("csv"))
        self.btn_export_xml = QPushButton("📄  Export XML", objectName="secondary")
        self.btn_export_xml.setEnabled(False); self.btn_export_xml.clicked.connect(lambda: self.export_report("xml"))
        self.btn_copy_json = QPushButton("📋  Copy JSON", objectName="secondary")
        self.btn_copy_json.setEnabled(False); self.btn_copy_json.clicked.connect(self.copy_json)
        
        self.btn_clear = QPushButton("🗑  Clear", objectName="danger"); self.btn_clear.clicked.connect(self.clear_tab)
        self.lbl_path = QLabel("No file loaded. Drag & Drop supported."); self.lbl_path.setStyleSheet("color:#8b949e;")
        
        self.sens_slider = QSlider(Qt.Horizontal); self.sens_slider.setRange(0, 100); self.sens_slider.setValue(self.sensitivity)
        self.sens_slider.setFixedWidth(100); self.sens_slider.valueChanged.connect(self.on_sens_changed)
        self.sens_label = QLabel(f"Sensitivity: {self.sensitivity}%")
        self.sens_label.setStyleSheet("color:#58a6ff; font-size:12px; font-weight:600; min-width: 100px;")
        
        ctrl.addWidget(self.btn_load); ctrl.addWidget(self.btn_analyze); ctrl.addWidget(self.chk_deep)
        ctrl.addWidget(self.btn_export_md); ctrl.addWidget(self.btn_export_pdf); ctrl.addWidget(self.btn_export_html); ctrl.addWidget(self.btn_export_csv); ctrl.addWidget(self.btn_export_xml); ctrl.addWidget(self.btn_copy_json); ctrl.addWidget(self.btn_clear)
        ctrl.addWidget(self.lbl_path, 1)
        ctrl.addWidget(self.sens_label); ctrl.addWidget(self.sens_slider)
        outer.addLayout(ctrl)

        if self.modality == "image":
            url_box = QGroupBox("Analyze from Web URL")
            url_layout = QHBoxLayout(url_box)
            self.url_input = QLineEdit(); self.url_input.setPlaceholderText("https://example.com/image.jpg")
            url_layout.addWidget(self.url_input)
            self.btn_load_url = QPushButton("🌐  Fetch & Analyze", objectName="secondary")
            self.btn_load_url.clicked.connect(self.load_from_url)
            url_layout.addWidget(self.btn_load_url)
            outer.addWidget(url_box)

        self.progress = QProgressBar(); self.progress.setValue(0); self.progress.setFormat("%v%  %m")
        self.progress.setVisible(False); outer.addWidget(self.progress)

        splitter = QSplitter(Qt.Horizontal)

        left = QWidget(); lv = QVBoxLayout(left); lv.setContentsMargins(0, 0, 0, 0)
        self.verdict_label = QLabel("Awaiting analysis", objectName="body")
        self.verdict_label.setAlignment(Qt.AlignCenter); self.verdict_label.setMinimumHeight(60)
        lv.addWidget(self.verdict_label)

        self.score_bar = QProgressBar(); self.score_bar.setRange(0, 100); self.score_bar.setTextVisible(True); self.score_bar.setFormat("0%")
        self.score_bar.setStyleSheet("QProgressBar { background: #0d1117; border: 1px solid #21262d; border-radius: 6px; text-align: center; color: white; font-weight: bold; } QProgressBar::chunk { background: #1f6feb; border-radius: 6px; }")
        self.score_bar.setFixedHeight(25); lv.addWidget(self.score_bar)

        self.chk_overlay = QCheckBox("Show Heatmap Overlay on Image")
        self.chk_overlay.setStyleSheet("color:#c9d1d9; font-weight:600;")
        self.chk_overlay.setVisible(False); self.chk_overlay.toggled.connect(self.toggle_overlay)
        lv.addWidget(self.chk_overlay)

        self.thumbnail_label = QLabel("Drag & Drop file here\n\nor click Load", objectName="drop_zone")
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        self.thumbnail_label.setStyleSheet("background:#0d1117; border:2px dashed #30363d; border-radius:8px; color:#8b949e;")
        self.thumbnail_label.setMinimumHeight(220); lv.addWidget(self.thumbnail_label, 1)
        splitter.addWidget(left)

        right = QWidget(); rv = QVBoxLayout(right); rv.setContentsMargins(0, 0, 0, 0)
        self.metrics_box = QGroupBox("Computed Metrics & File Hash"); QGridLayout(self.metrics_box).setSpacing(6)
        rv.addWidget(self.metrics_box)
        self.reasons_box = QGroupBox("Detected Anomalies / Signals")
        self.reasons_text = QTextEdit(); self.reasons_text.setReadOnly(True); self.reasons_text.setMaximumHeight(110)
        QVBoxLayout(self.reasons_box).addWidget(self.reasons_text); rv.addWidget(self.reasons_box)
        self.method_box = QGroupBox("Research Methodology")
        self.method_text = QTextEdit(); self.method_text.setReadOnly(True); self.method_text.setMaximumHeight(100)
        QVBoxLayout(self.method_box).addWidget(self.method_text); rv.addWidget(self.method_box)
        self.viz_label = QLabel("Visualization & Radar will appear here")
        self.viz_label.setAlignment(Qt.AlignCenter)
        self.viz_label.setStyleSheet("background:#0d1117; border:1px solid #21262d; border-radius:6px;")
        self.viz_label.setMinimumHeight(280); rv.addWidget(self.viz_label, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1); splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter, 1)

    def load_from_url(self):
        url = self.url_input.text().strip()
        if not url: return
        self.progress.setVisible(True); self.progress.setValue(10); self.progress.setFormat("Downloading image...")
        self.btn_load_url.setEnabled(False)
        self.url_worker = URLDownloadWorker(url)
        self.url_worker.finished.connect(self.on_url_downloaded)
        self.url_worker.error.connect(self.on_url_error)
        self.url_worker.start()

    def on_url_downloaded(self, path):
        self.progress.setVisible(False); self.btn_load_url.setEnabled(True)
        self.current_path = path
        self.lbl_path.setText(f"Loaded: {os.path.basename(path)} [URL]"); self.lbl_path.setStyleSheet("color:#3fb950;")
        self.btn_analyze.setEnabled(True); self.show_preview(path)

    def on_url_error(self, err):
        self.progress.setVisible(False); self.btn_load_url.setEnabled(True)
        QMessageBox.critical(self, "Download Error", f"Failed to fetch image:\n{err}")

    def on_sens_changed(self, val):
        self.sensitivity = val
        self.sens_label.setText(f"Sensitivity: {val}%")

    def load_file(self, file_filter):
        path, _ = QFileDialog.getOpenFileName(self, f"Load {self.modality}", "", file_filter)
        if path:
            self.current_path = path
            self.lbl_path.setText(f"Loaded: {os.path.basename(path)}"); self.lbl_path.setStyleSheet("color:#3fb950;")
            self.btn_analyze.setEnabled(True); self.show_preview(path)

    def show_preview(self, path):
        try:
            self.thumbnail_label.setText("Loading preview...")
            if self.modality == "image" and HAS_PIL:
                img = Image.open(path).convert("RGB"); img.thumbnail((400, 400))
                self.base_pixmap = pixmap_from_pil(img); self.thumbnail_label.setPixmap(self.base_pixmap)
            elif self.modality == "video" and HAS_CV2:
                cap = cv2.VideoCapture(path); ret, frame = cap.read(); cap.release()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); h, w, _ = rgb.shape
                    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
                    self.base_pixmap = QPixmap.fromImage(qimg).scaledToWidth(400, Qt.SmoothTransformation)
                    self.thumbnail_label.setPixmap(self.base_pixmap)
            elif self.modality == "audio":
                self.thumbnail_label.setText("🔊  Audio file loaded\n(Radar & Spectrogram shown after analysis)")
            elif self.modality == "text":
                with open(path, "r", encoding="utf-8", errors="ignore") as f: txt = f.read()[:500]
                self.thumbnail_label.setText(f"📄 Preview:\n\n{txt}"); self.thumbnail_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        except Exception as e: self.thumbnail_label.setText(f"Preview error: {e}")

    def get_custom_params(self): return self.custom_lexicon, self.custom_rules

    def clear_tab(self):
        self.current_path, self.last_result, self.base_pixmap = None, None, None
        self.lbl_path.setText("No file loaded. Drag & Drop supported."); self.lbl_path.setStyleSheet("color:#8b949e;")
        self.btn_analyze.setEnabled(False); self.btn_export_md.setEnabled(False); self.btn_export_pdf.setEnabled(False); self.btn_export_html.setEnabled(False); self.btn_export_csv.setEnabled(False); self.btn_export_xml.setEnabled(False); self.btn_copy_json.setEnabled(False)
        self.chk_overlay.setVisible(False); self.chk_overlay.setChecked(False)
        self.verdict_label.setText("Awaiting analysis")
        self.verdict_label.setStyleSheet("color:#c9d1d9; background:#0d1117; border:1px solid #21262d; border-radius:6px; padding:8px;")
        self.score_bar.setValue(0); self.score_bar.setFormat("0%")
        self.thumbnail_label.setText("Drag & Drop file here\n\nor click Load"); self.thumbnail_label.setPixmap(QPixmap())
        self.reasons_text.setPlainText(""); self.method_text.setPlainText("")

    def run_detection(self):
        if not self.current_path: return
        self.progress.setVisible(True); self.progress.setValue(0)
        self.btn_analyze.setEnabled(False); self.btn_load.setEnabled(False)
        self.btn_export_md.setEnabled(False); self.btn_export_pdf.setEnabled(False); self.btn_export_html.setEnabled(False); self.btn_export_csv.setEnabled(False); self.btn_export_xml.setEnabled(False); self.btn_copy_json.setEnabled(False)
        self.deep_scan = self.chk_deep.isChecked()
        lexicon, rules = self.get_custom_params()
        self.worker = DetectionWorker(self.modality, self.current_path, self.sensitivity, self.deep_scan, lexicon, rules)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_result.connect(self.on_result)
        self.worker.start()

    def on_progress(self, val, msg): self.progress.setValue(val); self.progress.setFormat(f"{val}%  {msg}")

    def export_report(self, fmt):
        if not self.last_result: return
        ext = fmt
        default_name = f"deepguard_report_{self.modality}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"
        path, _ = QFileDialog.getSaveFileName(self, "Save Report", default_name, f"{ext.upper()} Files (*.{ext})")
        if path:
            try:
                res = self.last_result
                custody_hash = generate_chain_of_custody_hash(res)
                if fmt == "md":
                    md = f"# DeepGuard Analysis Report\n\n**File:** `{res.get('file_name', 'N/A')}`\n\n**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n**Verdict:** {res.get('verdict', 'N/A')}\n\n**AI Probability:** {res.get('score', 0)*100:.1f}% ± {res.get('margin_of_error', 0)*100:.1f}%\n\n**File SHA-256:** `{res.get('file_hash', 'N/A')}`\n\n## Metrics\n\n"
                    for k, v in res.get("metrics", {}).items(): md += f"- **{k}:** {v}\n"
                    md += "\n## Detected Anomalies\n\n"
                    for r in res.get("reasons", []): md += f"- {r}\n"
                    md += f"\n## Methodology\n\n{res.get('methodology', 'N/A')}\n\n"
                    md += f"\n---\n**Chain of Custody Hash:** `{custody_hash}`\n"
                    if res.get("viz") and HAS_MPL: md += f"![Visualization](data:image/png;base64,{fig_to_base64(res['viz'])})\n"
                    with open(path, "w") as f: f.write(md)
                elif fmt == "pdf":
                    html = f"<div style='font-family: Arial; color: #333;'><h1 style='color:#1f6feb;'>DeepGuard Analysis Report</h1><p><b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p><p><b>File:</b> {res.get('file_name', 'N/A')}</p><p><b>Verdict:** <span style='color:{"#f85149" if res.get("score",0)>res.get("threshold",0.5) else "#3fb950"}; font-weight:bold;'>{res.get('verdict', 'N/A')}</span></p><p><b>AI Probability:</b> {res.get('score', 0)*100:.1f}% ± {res.get('margin_of_error', 0)*100:.1f}%</p><p><b>File SHA-256:</b> <code>{res.get('file_hash', 'N/A')}</code></p><h2 style='color:#1f6feb;'>Metrics</h2><ul>"
                    for k, v in res.get("metrics", {}).items(): html += f"<li><b>{k}:</b> {v}</li>"
                    html += "</ul><h2 style='color:#1f6feb;'>Detected Anomalies</h2><ul>"
                    for r in res.get("reasons", []): html += f"<li>{r}</li>"
                    html += f"</ul><h2 style='color:#1f6feb;'>Methodology</h2><p>{res.get('methodology', 'N/A')}</p>"
                    if res.get("viz") and HAS_MPL: html += f"<h2 style='color:#1f6feb;'>Visualizations</h2><img src='data:image/png;base64,{fig_to_base64(res['viz'])}' width='600'/>"
                    html += f"<hr><p><b>Chain of Custody Hash:</b> <code>{custody_hash}</code></p>"
                    html += "</div>"
                    doc = QTextDocument(); doc.setHtml(html)
                    printer = QPrinter(QPrinter.HighResolution); printer.setOutputFormat(QPrinter.PdfFormat); printer.setOutputFileName(path)
                    doc.print_(printer)
                elif fmt == "html":
                    html = "<html><head><style>body{font-family:Arial;background:#f4f4f4;color:#333;} .container{max-width:800px;margin:20px auto;background:white;padding:20px;border-radius:8px;box-shadow:0 0 10px rgba(0,0,0,0.1);} h1{color:#1f6feb;border-bottom:2px solid #1f6feb;padding-bottom:10px;} h2{color:#24292f;margin-top:20px;} .metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;} .metric-item{background:#f6f8fa;padding:10px;border-radius:4px;} .metric-title{color:#57606a;font-size:12px;font-weight:bold;} .metric-val{color:#24292f;font-size:16px;} .verdict{padding:10px;border-radius:4px;font-weight:bold;text-align:center;font-size:18px;margin:20px 0;} .fake{background:#ffeaea;color:#d9534f;} .real{background:#eaffea;color:#5cb85c;} ul{background:#f6f8fa;padding:15px 15px 15px 35px;border-radius:4px;} img{max-width:100%;border-radius:4px;margin-top:10px;} .footer{margin-top:20px;border-top:1px solid #ddd;padding-top:10px;font-size:12px;color:#57606a;}</style></head><body><div class='container'>"
                    html += f"<h1>DeepGuard Analysis Report</h1>"
                    html += f"<p><b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
                    html += f"<p><b>File:</b> {res.get('file_name', 'N/A')}</p>"
                    cls = "fake" if res.get("score",0)>res.get("threshold",0.5) else "real"
                    html += f"<div class='verdict {cls}'>Verdict: {res.get('verdict', 'N/A')} ({res.get('score', 0)*100:.1f}% ± {res.get('margin_of_error', 0)*100:.1f}%)</div>"
                    html += f"<p><b>File SHA-256:</b> <code>{res.get('file_hash', 'N/A')}</code></p>"
                    html += "<h2>Metrics</h2><div class='metric-grid'>"
                    for k, v in res.get("metrics", {}).items(): html += f"<div class='metric-item'><div class='metric-title'>{k}</div><div class='metric-val'>{v}</div></div>"
                    html += "</div><h2>Detected Anomalies</h2><ul>"
                    for r in res.get("reasons", []): html += f"<li>{r}</li>"
                    html += "</ul>"
                    html += f"<h2>Methodology</h2><p>{res.get('methodology', 'N/A')}</p>"
                    if res.get("viz") and HAS_MPL: html += f"<h2>Visualizations</h2><img src='data:image/png;base64,{fig_to_base64(res['viz'])}'/>"
                    html += f"<div class='footer'><b>Chain of Custody Hash:</b> <code>{custody_hash}</code></div>"
                    html += "</div></body></html>"
                    with open(path, "w", encoding="utf-8") as f: f.write(html)
                elif fmt == "csv":
                    with open(path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(["Field", "Value"])
                        writer.writerow(["File Name", res.get('file_name', '')])
                        writer.writerow(["Verdict", res.get('verdict', '')])
                        writer.writerow(["AI Probability", f"{res.get('score', 0)*100:.1f}%"])
                        writer.writerow(["File SHA-256", res.get('file_hash', '')])
                        writer.writerow(["Custody Hash", custody_hash])
                        writer.writerow([])
                        writer.writerow(["Metrics"])
                        for k, v in res.get("metrics", {}).items(): writer.writerow([k, v])
                        writer.writerow([])
                        writer.writerow(["Anomalies"])
                        for r in res.get("reasons", []): writer.writerow([r])
                elif fmt == "xml":
                    root = ET.Element("DeepGuardReport")
                    ET.SubElement(root, "File", name=res.get('file_name', ''))
                    ET.SubElement(root, "Verdict").text = res.get('verdict', '')
                    ET.SubElement(root, "AIProbability").text = f"{res.get('score', 0)*100:.1f}%"
                    ET.SubElement(root, "FileSHA256").text = res.get('file_hash', '')
                    ET.SubElement(root, "CustodyHash").text = custody_hash
                    metrics_elem = ET.SubElement(root, "Metrics")
                    for k, v in res.get("metrics", {}).items():
                        ET.SubElement(metrics_elem, "Metric", name=k).text = str(v)
                    anomalies_elem = ET.SubElement(root, "Anomalies")
                    for r in res.get("reasons", []):
                        ET.SubElement(anomalies_elem, "Anomaly").text = r
                    ET.SubElement(root, "Methodology").text = res.get('methodology', '')
                    tree = ET.ElementTree(root)
                    tree.write(path, encoding="utf-8", xml_declaration=True)
                QMessageBox.information(self, "Export Success", f"Report saved to:\n{path}")
            except Exception as e: QMessageBox.critical(self, "Export Error", str(e))

    def copy_json(self):
        if not self.last_result: return
        cb = QApplication.clipboard()
        clean_res = {k: v for k, v in self.last_result.items() if k not in ["viz", "thumbnail", "thumbnails", "ela_heatmap", "faces", "score_components"]}
        cb.setText(json.dumps(clean_res, indent=4))
        self.statusBar().showMessage("JSON copied to clipboard!", 3000)

    def on_result(self, res):
        self.progress.setVisible(False); self.btn_analyze.setEnabled(True); self.btn_load.setEnabled(True)
        if "error" in res:
            self.verdict_label.setText(f"⚠️  Error: {res['error']}"); self.verdict_label.setStyleSheet("color:#f85149; font-weight:700; font-size:14px;")
            return

        self.last_result = res; self.result_logged.emit(res)
        self.btn_export_md.setEnabled(True); self.btn_export_pdf.setEnabled(True); self.btn_export_html.setEnabled(True); self.btn_export_csv.setEnabled(True); self.btn_export_xml.setEnabled(True); self.btn_copy_json.setEnabled(True)
        score, verdict, threshold = res.get("score", 0), res.get("verdict", "Unknown"), res.get("threshold", 0.5)
        margin = res.get("margin_of_error", 0.0)
        self.score_bar.setValue(int(score * 100)); self.score_bar.setFormat(f"{score*100:.1f}% ± {margin*100:.1f}%")
        
        if score > threshold:
            self.verdict_label.setText(f"🚨  {verdict}")
            self.verdict_label.setStyleSheet("color:#f85149; font-size:18px; font-weight:700; background:#2a0f12; border:1px solid #f85149; border-radius:6px; padding:8px;")
            self.score_bar.setStyleSheet("QProgressBar { background: #0d1117; border: 1px solid #21262d; border-radius: 6px; text-align: center; color: white; font-weight: bold; } QProgressBar::chunk { background: #f85149; border-radius: 6px; }")
        else:
            self.verdict_label.setText(f"✅  {verdict}")
            self.verdict_label.setStyleSheet("color:#3fb950; font-size:18px; font-weight:700; background:#0f2a18; border:1px solid #3fb950; border-radius:6px; padding:8px;")
            self.score_bar.setStyleSheet("QProgressBar { background: #0d1117; border: 1px solid #21262d; border-radius: 6px; text-align: center; color: white; font-weight: bold; } QProgressBar::chunk { background: #3fb950; border-radius: 6px; }")

        while self.metrics_box.layout().count():
            item = self.metrics_box.layout().takeAt(0)
            if item.widget(): item.widget().deleteLater()
        metrics = res.get("metrics", {}); metrics["File SHA-256"] = res.get("file_hash", "N/A")
        for i, (k, v) in enumerate(metrics.items()):
            lbl = QLabel(f"<b style='color:#8b949e'>{k}</b><br><span style='color:#c9d1d9; font-size:14px'>{v}</span>")
            lbl.setStyleSheet("padding:4px;"); lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.metrics_box.layout().addWidget(lbl, i // 2, i % 2)

        self.reasons_text.setPlainText("\n".join(f"• {r}" for r in res.get("reasons", [])) if res.get("reasons") else "No strong anomaly signals detected.")
        self.method_text.setPlainText(res.get("methodology", ""))

        viz = res.get("viz")
        if viz is not None and HAS_MPL:
            canvas = FigureCanvas(viz); canvas.setStyleSheet("background:#0d1117;")
            parent = self.viz_label.parentWidget(); layout = parent.layout()
            idx = layout.indexOf(self.viz_label); self.viz_label.setParent(None); self.viz_label = canvas
            layout.insertWidget(idx, canvas)
        elif "thumbnail" in res:
            self.base_pixmap = pixmap_from_pil(res["thumbnail"], max_w=400)
            self.thumbnail_label.setPixmap(self.base_pixmap)
        elif "thumbnails" in res:
            thumbs = res["thumbnails"]; combined = QWidget(); ch = QHBoxLayout(combined); ch.setContentsMargins(0, 0, 0, 0)
            for t in thumbs:
                lbl = QLabel(); h, w, _ = t.shape
                qimg = QImage(t.data, w, h, w * 3, QImage.Format_RGB888)
                lbl.setPixmap(QPixmap.fromImage(qimg).scaledToHeight(140, Qt.SmoothTransformation)); ch.addWidget(lbl)
            self.thumbnail_label.setParent(None); self.thumbnail_label = combined; parent.layout().addWidget(combined)

        if self.modality == "image" and "ela_heatmap" in res:
            self.chk_overlay.setVisible(True)

    def toggle_overlay(self, checked):
        if not self.base_pixmap or not self.last_result: return
        if checked:
            heatmap = self.last_result.get("ela_heatmap")
            faces = self.last_result.get("faces", [])
            if heatmap is None: return
            hm_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
            hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
            qimg_hm = QImage(hm_color.data, hm_color.shape[1], hm_color.shape[0], hm_color.shape[2], QImage.Format_RGB888)
            hm_pixmap = QPixmap.fromImage(qimg_hm).scaled(self.base_pixmap.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            
            overlay_pixmap = QPixmap(self.base_pixmap.size()); overlay_pixmap.fill(Qt.transparent)
            painter = QPainter(overlay_pixmap)
            painter.drawPixmap(0, 0, self.base_pixmap)
            painter.setOpacity(0.6); painter.drawPixmap(0, 0, hm_pixmap)
            painter.setOpacity(1.0); pen = QPen(QColor(0, 255, 255), 2); painter.setPen(pen)
            for x, y, fw, fh in faces:
                scale_x = self.base_pixmap.width() / heatmap.shape[1]
                scale_y = self.base_pixmap.height() / heatmap.shape[0]
                painter.drawRect(int(x*scale_x), int(y*scale_y), int(fw*scale_x), int(fh*scale_y))
            painter.end()
            self.thumbnail_label.setPixmap(overlay_pixmap)
        else: self.thumbnail_label.setPixmap(self.base_pixmap)

    def load_from_history(self, res):
        self.current_path = res.get("file_name", "")
        self.lbl_path.setText(f"Loaded: {os.path.basename(self.current_path)}"); self.lbl_path.setStyleSheet("color:#3fb950;")
        self.btn_analyze.setEnabled(True); self.on_result(res)


# ===========================================================================
#  TEXT TAB
# ===========================================================================
class TextDetectionTab(DetectionTab):
    def __init__(self):
        self.text_edit = None; self.lexicon_input = None
        super().__init__("text", "Text Document", "Text Files (*.txt *.md);;All Files (*)")

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls() or event.mimeData().hasText(): event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            path = event.mimeData().urls()[0].toLocalFile()
            if path:
                self.current_path = path
                self.lbl_path.setText(f"Loaded: {os.path.basename(path)}"); self.lbl_path.setStyleSheet("color:#3fb950;")
                self.btn_analyze.setEnabled(True); self.show_preview(path)
        elif event.mimeData().hasText():
            self.text_edit.setPlainText(event.mimeData().text())
            self.lbl_path.setText("Text loaded from drag & drop"); self.lbl_path.setStyleSheet("color:#3fb950;")

    def build_ui(self, accept_label, file_filter):
        outer = QVBoxLayout(self); outer.setSpacing(12)
        outer.addWidget(QLabel("TEXT DETECTION", objectName="title"))
        outer.addWidget(QLabel("Paste text, drag & drop, or load a file. Min 50 words recommended.", objectName="subtitle"))

        ctrl = QHBoxLayout()
        self.btn_load = QPushButton("📁  Load Text File"); self.btn_load.clicked.connect(lambda: self.load_file(file_filter))
        self.btn_analyze = QPushButton("🔬  Run Detection", objectName="secondary"); self.btn_analyze.clicked.connect(self.run_detection)
        self.chk_deep = QCheckBox("Deep Scan"); self.chk_deep.setStyleSheet("color:#c9d1d9; font-weight:600;")
        self.btn_export_md = QPushButton("📄  Export MD", objectName="secondary"); self.btn_export_md.setEnabled(False); self.btn_export_md.clicked.connect(lambda: self.export_report("md"))
        self.btn_export_pdf = QPushButton("📄  Export PDF", objectName="secondary"); self.btn_export_pdf.setEnabled(False); self.btn_export_pdf.clicked.connect(lambda: self.export_report("pdf"))
        self.btn_export_html = QPushButton("📄  Export HTML", objectName="secondary"); self.btn_export_html.setEnabled(False); self.btn_export_html.clicked.connect(lambda: self.export_report("html"))
        self.btn_export_csv = QPushButton("📄  Export CSV", objectName="secondary"); self.btn_export_csv.setEnabled(False); self.btn_export_csv.clicked.connect(lambda: self.export_report("csv"))
        self.btn_export_xml = QPushButton("📄  Export XML", objectName="secondary"); self.btn_export_xml.setEnabled(False); self.btn_export_xml.clicked.connect(lambda: self.export_report("xml"))
        self.btn_copy_json = QPushButton("📋  Copy JSON", objectName="secondary"); self.btn_copy_json.setEnabled(False); self.btn_copy_json.clicked.connect(self.copy_json)
        self.btn_clear = QPushButton("🗑  Clear", objectName="danger"); self.btn_clear.clicked.connect(self.clear_tab)
        self.lbl_path = QLabel("Drag & Drop supported."); self.lbl_path.setStyleSheet("color:#8b949e;")
        
        self.sens_slider = QSlider(Qt.Horizontal); self.sens_slider.setRange(0, 100); self.sens_slider.setValue(self.sensitivity)
        self.sens_slider.setFixedWidth(100); self.sens_slider.valueChanged.connect(self.on_sens_changed)
        self.sens_label = QLabel(f"Sensitivity: {self.sensitivity}%")
        self.sens_label.setStyleSheet("color:#58a6ff; font-size:12px; font-weight:600; min-width: 100px;")
        
        ctrl.addWidget(self.btn_load); ctrl.addWidget(self.btn_analyze); ctrl.addWidget(self.chk_deep)
        ctrl.addWidget(self.btn_export_md); ctrl.addWidget(self.btn_export_pdf); ctrl.addWidget(self.btn_export_html); ctrl.addWidget(self.btn_export_csv); ctrl.addWidget(self.btn_export_xml); ctrl.addWidget(self.btn_copy_json); ctrl.addWidget(self.btn_clear)
        ctrl.addWidget(self.lbl_path, 1)
        ctrl.addWidget(self.sens_label); ctrl.addWidget(self.sens_slider)
        outer.addLayout(ctrl)

        lexicon_box = QGroupBox("Custom LLM Lexicon (Comma-Separated)")
        self.lexicon_input = QLineEdit(); self.lexicon_input.setPlaceholderText("e.g., underscore, delve, tapestry, custom_word")
        QHBoxLayout(lexicon_box).addWidget(self.lexicon_input); outer.addWidget(lexicon_box)

        self.text_edit = QTextEdit(); self.text_edit.setPlaceholderText("Paste or drop the text you want to analyze here...")
        self.text_edit.setMinimumHeight(140); outer.addWidget(QLabel("Input Text:")); outer.addWidget(self.text_edit)

        self.progress = QProgressBar(); self.progress.setValue(0); self.progress.setFormat("%v%  %m")
        self.progress.setVisible(False); outer.addWidget(self.progress)

        splitter = QSplitter(Qt.Vertical)
        results = QWidget(); rv = QVBoxLayout(results); rv.setContentsMargins(0, 0, 0, 0)
        self.verdict_label = QLabel("Awaiting analysis", objectName="body"); self.verdict_label.setAlignment(Qt.AlignCenter); self.verdict_label.setMinimumHeight(50)
        rv.addWidget(self.verdict_label)
        self.score_bar = QProgressBar(); self.score_bar.setRange(0, 100); self.score_bar.setFormat("0%")
        self.score_bar.setFixedHeight(25); rv.addWidget(self.score_bar)

        self.metrics_box = QGroupBox("Computed Metrics & File Hash"); QGridLayout(self.metrics_box).setSpacing(6); rv.addWidget(self.metrics_box)
        self.reasons_box = QGroupBox("Detected Anomalies / Signals")
        self.reasons_text = QTextEdit(); self.reasons_text.setReadOnly(True); self.reasons_text.setMaximumHeight(100)
        QVBoxLayout(self.reasons_box).addWidget(self.reasons_text); rv.addWidget(self.reasons_box)
        self.method_box = QGroupBox("Research Methodology")
        self.method_text = QTextEdit(); self.method_text.setReadOnly(True); self.method_text.setMaximumHeight(90)
        QVBoxLayout(self.method_box).addWidget(self.method_text); rv.addWidget(self.method_box)
        splitter.addWidget(results)

        self.viz_label = QLabel("Visualization & Radar will appear here")
        self.viz_label.setAlignment(Qt.AlignCenter)
        self.viz_label.setStyleSheet("background:#0d1117; border:1px solid #21262d; border-radius:6px;")
        self.viz_label.setMinimumHeight(240); splitter.addWidget(self.viz_label)
        splitter.setStretchFactor(0, 2); splitter.setStretchFactor(1, 1); outer.addWidget(splitter, 1)

    def run_detection(self):
        text = self.text_edit.toPlainText().strip()
        if len(text) < 20: return QMessageBox.warning(self, "Too short", "Please enter at least 20 characters.")
        self.progress.setVisible(True); self.progress.setValue(0)
        self.btn_analyze.setEnabled(False); self.btn_export_md.setEnabled(False); self.btn_export_pdf.setEnabled(False); self.btn_export_html.setEnabled(False); self.btn_export_csv.setEnabled(False); self.btn_export_xml.setEnabled(False); self.btn_copy_json.setEnabled(False)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tf: tf.write(text); tmp = tf.name
        self.current_path = tmp; self.deep_scan = self.chk_deep.isChecked()
        lexicon = self.lexicon_input.text().strip()
        rules = QApplication.instance().property("CustomRules") if QApplication.instance().property("CustomRules") else ""
        self.worker = DetectionWorker("text", tmp, self.sensitivity, self.deep_scan, lexicon, rules)
        self.worker.progress.connect(self.on_progress); self.worker.finished_result.connect(self.on_result)
        self.worker.start()


# ===========================================================================
#  COMPARISON TAB
# ===========================================================================
class ComparisonTab(QWidget):
    def __init__(self, sensitivity=50):
        super().__init__(); self.sensitivity = sensitivity; self.res_a, self.res_b = None, None; self.build_ui()

    def set_sensitivity(self, val): self.sensitivity = val

    def build_ui(self):
        outer = QVBoxLayout(self); outer.setSpacing(12)
        outer.addWidget(QLabel("⚖️  SIDE-BY-SIDE COMPARISON", objectName="title"))
        outer.addWidget(QLabel("Analyze two files (e.g., a known real baseline vs a suspicious file) to overlay their anomaly radars.", objectName="subtitle"))

        splitter = QSplitter(Qt.Horizontal)
        left_widget = QWidget(); lv = QVBoxLayout(left_widget)
        lv.addWidget(QLabel("<h3 style='color:#58a6ff'>File A (Baseline)</h3>"))
        self.btn_load_a = QPushButton("📁  Load File A"); self.btn_load_a.clicked.connect=lambda: self.load_file("A")
        self.btn_run_a = QPushButton("🔬  Analyze A", objectName="secondary"); self.btn_run_a.setEnabled(False); self.btn_run_a.clicked.connect=lambda: self.run_detection("A")
        lv.addWidget(self.btn_load_a); lv.addWidget(self.btn_run_a)
        self.lbl_path_a = QLabel("No file loaded."); self.lbl_path_a.setStyleSheet("color:#8b949e;"); lv.addWidget(self.lbl_path_a)
        self.lbl_verdict_a = QLabel("Awaiting analysis", objectName="body"); self.lbl_verdict_a.setAlignment(Qt.AlignCenter); self.lbl_verdict_a.setMinimumHeight(50)
        lv.addWidget(self.lbl_verdict_a)
        self.lbl_preview_a = QLabel("Drop file here", objectName="drop_zone"); self.lbl_preview_a.setAlignment(Qt.AlignCenter)
        self.lbl_preview_a.setStyleSheet("background:#0d1117; border:2px dashed #30363d; border-radius:8px; color:#8b949e;")
        lv.addWidget(self.lbl_preview_a, 1); splitter.addWidget(left_widget)

        right_widget = QWidget(); rv = QVBoxLayout(right_widget)
        rv.addWidget(QLabel("<h3 style='color:#f85149'>File B (Suspicious)</h3>"))
        self.btn_load_b = QPushButton("📁  Load File B"); self.btn_load_b.clicked.connect=lambda: self.load_file("B")
        self.btn_run_b = QPushButton("🔬  Analyze B", objectName="secondary"); self.btn_run_b.setEnabled(False); self.btn_run_b.clicked.connect=lambda: self.run_detection("B")
        rv.addWidget(self.btn_load_b); rv.addWidget(self.btn_run_b)
        self.lbl_path_b = QLabel("No file loaded."); self.lbl_path_b.setStyleSheet("color:#8b949e;"); rv.addWidget(self.lbl_path_b)
        self.lbl_verdict_b = QLabel("Awaiting analysis", objectName="body"); self.lbl_verdict_b.setAlignment(Qt.AlignCenter); self.lbl_verdict_b.setMinimumHeight(50)
        rv.addWidget(self.lbl_verdict_b)
        self.lbl_preview_b = QLabel("Drop file here", objectName="drop_zone"); self.lbl_preview_b.setAlignment(Qt.AlignCenter)
        self.lbl_preview_b.setStyleSheet("background:#0d1117; border:2px dashed #30363d; border-radius:8px; color:#8b949e;")
        rv.addWidget(self.lbl_preview_b, 1); splitter.addWidget(right_widget)
        outer.addWidget(splitter, 1)

        self.viz_label = QLabel("Overlay Radar will appear here after both are analyzed")
        self.viz_label.setAlignment(Qt.AlignCenter)
        self.viz_label.setStyleSheet("background:#0d1117; border:1px solid #21262d; border-radius:6px;")
        self.viz_label.setMinimumHeight(300); outer.addWidget(self.viz_label, 1)
        self.path_a, self.path_b, self.modality_a, self.modality_b = None, None, None, None

    def load_file(self, slot):
        path, _ = QFileDialog.getOpenFileName(self, f"Load File {slot}", "", "All Files (*)")
        if not path: return
        ext = os.path.splitext(path)[1].lower(); modality = None
        if ext in [".wav", ".mp3"]: modality = "audio"
        elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]: modality = "image"
        elif ext in [".mp4", ".avi", ".mov", ".mkv"]: modality = "video"
        elif ext in [".txt", ".md"]: modality = "text"
        else: return QMessageBox.warning(self, "Invalid File", "Unsupported file type for comparison.")

        if slot == "A":
            self.path_a, self.modality_a = path, modality
            self.lbl_path_a.setText(f"Loaded: {os.path.basename(path)} [{modality}]"); self.lbl_path_a.setStyleSheet("color:#3fb950;")
            self.btn_run_a.setEnabled(True)
        else:
            self.path_b, self.modality_b = path, modality
            self.lbl_path_b.setText(f"Loaded: {os.path.basename(path)} [{modality}]"); self.lbl_path_b.setStyleSheet("color:#3fb950;")
            self.btn_run_b.setEnabled(True)

    def run_detection(self, slot):
        path = self.path_a if slot == "A" else self.path_b
        modality = self.modality_a if slot == "A" else self.modality_b
        if not path or not modality: return
        worker = DetectionWorker(modality, path, self.sensitivity, False, "")
        worker.finished_result.connect(lambda res: self.on_result(slot, res))
        worker.start()

    def on_result(self, slot, res):
        if "error" in res: return QMessageBox.critical(self, "Error", res["error"])
        if slot == "A":
            self.res_a = res
            self.lbl_verdict_a.setText(f"{res.get('score',0)*100:.1f}% {res.get('verdict','')}")
            self.lbl_verdict_a.setStyleSheet("color:#58a6ff; font-weight:700; background:#0d1117; border:1px solid #58a6ff; border-radius:6px; padding:8px;")
        else:
            self.res_b = res
            self.lbl_verdict_b.setText(f"{res.get('score',0)*100:.1f}% {res.get('verdict','')}")
            self.lbl_verdict_b.setStyleSheet("color:#f85149; font-weight:700; background:#0d1117; border:1px solid #f85149; border-radius:6px; padding:8px;")
        if self.res_a and self.res_b: self.draw_overlay()

    def draw_overlay(self):
        if not HAS_MPL: return
        radar_a, radar_b = self.res_a.get("radar", {}), self.res_b.get("radar", {})
        keys = list(dict.fromkeys(list(radar_a.keys()) + list(radar_b.keys())))
        stats_a, stats_b = [radar_a.get(k, 0) for k in keys], [radar_b.get(k, 0) for k in keys]
        angles = np.linspace(0, 2*np.pi, len(keys), endpoint=False).tolist()
        stats_a += stats_a[:1]; angles_a = angles + angles[:1]
        stats_b += stats_b[:1]; angles_b = angles + angles[:1]

        fig = Figure(figsize=(8, 4), facecolor="#0d1117" if QApplication.instance().property("Theme") == "dark" else "#ffffff")
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles_a, stats_a, color="#58a6ff", linewidth=2, label="File A"); ax.fill(angles_a, stats_a, color="#58a6ff", alpha=0.2)
        ax.plot(angles_b, stats_b, color="#f85149", linewidth=2, label="File B"); ax.fill(angles_b, stats_b, color="#f85149", alpha=0.2)
        ax.set_xticks(angles); ax.set_xticklabels(keys, color="#c9d1d9" if QApplication.instance().property("Theme") == "dark" else "#24292f", fontsize=9)
        ax.set_title("Anomaly Radar Overlay", color="#c9d1d9" if QApplication.instance().property("Theme") == "dark" else "#24292f", fontsize=12, pad=20); ax.grid(color="#30363d")
        ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", loc="upper right"); fig.tight_layout()

        canvas = FigureCanvas(fig); canvas.setStyleSheet("background:#0d1117;")
        parent = self.viz_label.parentWidget(); layout = parent.layout()
        idx = layout.indexOf(self.viz_label); self.viz_label.setParent(None); self.viz_label = canvas
        layout.insertWidget(idx, canvas)


# ===========================================================================
#  BATCH PROCESSING TAB (With Live Search)
# ===========================================================================
class BatchTab(QWidget):
    def __init__(self):
        super().__init__(); self.worker = None; self.build_ui()

    def build_ui(self):
        outer = QVBoxLayout(self); outer.setSpacing(12)
        outer.addWidget(QLabel("BATCH PROCESSING", objectName="title"))
        outer.addWidget(QLabel("Select a folder. DeepGuard will auto-detect file types, analyze them concurrently, and output a report.", objectName="subtitle"))

        ctrl = QHBoxLayout()
        self.btn_load = QPushButton("📁  Select Directory"); self.btn_load.clicked.connect(self.load_dir)
        self.btn_sort = QPushButton("🔄  Auto-Sort by Verdict", objectName="secondary"); self.btn_sort.clicked.connect(self.auto_sort)
        self.btn_export_csv = QPushButton("📄  Export CSV", objectName="secondary"); self.btn_export_csv.setEnabled(False); self.btn_export_csv.clicked.connect(lambda: self.export_data("csv"))
        self.btn_export_html = QPushButton("📄  Export HTML", objectName="secondary"); self.btn_export_html.setEnabled(False); self.btn_export_html.clicked.connect(lambda: self.export_data("html"))
        self.btn_export_jsonl = QPushButton("📄  Export JSONL", objectName="secondary"); self.btn_export_jsonl.setEnabled(False); self.btn_export_jsonl.clicked.connect(lambda: self.export_data("jsonl"))
        self.btn_clear = QPushButton("🗑  Clear Table", objectName="danger"); self.btn_clear.clicked.connect(self.clear_table)
        ctrl.addWidget(self.btn_load); ctrl.addWidget(self.btn_sort); ctrl.addWidget(self.btn_export_csv); ctrl.addWidget(self.btn_export_html); ctrl.addWidget(self.btn_export_jsonl); ctrl.addWidget(self.btn_clear); ctrl.addStretch()
        outer.addLayout(ctrl)

        self.lbl_path = QLabel("No directory selected."); self.lbl_path.setStyleSheet("color:#8b949e;"); outer.addWidget(self.lbl_path)
        
        depth_box = QHBoxLayout()
        depth_box.addWidget(QLabel("Max Subfolder Depth:"))
        self.spin_depth = QSpinBox(); self.spin_depth.setRange(0, 20); self.spin_depth.setValue(10)
        depth_box.addWidget(self.spin_depth); depth_box.addStretch()
        outer.addLayout(depth_box)
        
        self.progress = QProgressBar(); self.progress.setValue(0); outer.addWidget(self.progress)

        # NEW: Live Search Box
        search_box = QHBoxLayout()
        search_box.addWidget(QLabel("Filter:"))
        self.search_input = QLineEdit(); self.search_input.setPlaceholderText("Filter results by filename or verdict...")
        self.search_input.textChanged.connect(self.filter_table)
        search_box.addWidget(self.search_input)
        outer.addLayout(search_box)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["File Name", "Modality", "AI Probability", "Verdict", "Key Reasons"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers); outer.addWidget(self.table, 1)
        self.results_data = []

    def filter_table(self, text):
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item:
                self.table.setRowHidden(row, text.lower() not in item.text().lower())

    def load_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory for Batch Analysis")
        if dir_path:
            self.lbl_path.setText(f"Scanned: {dir_path}"); self.lbl_path.setStyleSheet("color:#3fb950;")
            self.table.setRowCount(0); self.results_data = []; files = []
            valid_exts = {".wav": "audio", ".mp3": "audio", ".flac": "audio", ".png": "image", ".jpg": "image", ".jpeg": "image", ".bmp": "image", ".webp": "image", ".mp4": "video", ".avi": "video", ".mov": "video", ".mkv": "video", ".txt": "text", ".md": "text"}
            
            max_depth = self.spin_depth.value()
            for root, dirs, fnames in os.walk(dir_path):
                if max_depth == 0 and root != dir_path: continue
                if max_depth > 0:
                    rel_path = os.path.relpath(root, dir_path)
                    if rel_path != "." and rel_path.count(os.sep) >= max_depth:
                        dirs[:] = []
                        continue
                
                for f in fnames:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in valid_exts: files.append((os.path.join(root, f), valid_exts[ext]))
            
            if not files: return
            self.btn_load.setEnabled(False)
            self.worker = BatchWorker(files)
            self.worker.progress.connect(self.on_progress); self.worker.row_ready.connect(self.add_row)
            self.worker.finished_batch.connect(self.on_finished); self.worker.start()

    def add_row(self, data):
        self.results_data.append(data)
        row = self.table.rowCount(); self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(data.get("File", "")))
        self.table.setItem(row, 1, QTableWidgetItem(data.get("Modality", "").upper()))
        score_item, verdict_item = QTableWidgetItem(data.get("AI Probability", "")), QTableWidgetItem(data.get("Verdict", ""))
        if data.get("Verdict") == "AI-Generated":
            score_item.setForeground(QColor("#f85149")); verdict_item.setForeground(QColor("#f85149"))
        else:
            score_item.setForeground(QColor("#3fb950")); verdict_item.setForeground(QColor("#3fb950"))
        self.table.setItem(row, 2, score_item); self.table.setItem(row, 3, verdict_item)
        self.table.setItem(row, 4, QTableWidgetItem(data.get("Key Reasons", "")))

    def auto_sort(self):
        self.results_data.sort(key=lambda x: 0 if "AI" in x.get("Verdict", "") else 1 if "Authentic" in x.get("Verdict", "") else 2)
        self.table.setRowCount(0)
        for data in self.results_data:
            self.add_row(data)

    def on_progress(self, val, msg): self.progress.setValue(val); self.progress.setFormat(f"{val}%  {msg}")
    def on_finished(self):
        self.btn_load.setEnabled(True); self.btn_export_csv.setEnabled(True); self.btn_export_html.setEnabled(True); self.btn_export_jsonl.setEnabled(True)
        QMessageBox.information(self, "Batch Complete", "Batch analysis finished.")
        
    def export_data(self, fmt):
        if not self.results_data: return
        ext = fmt
        default_name = f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"
        path, _ = QFileDialog.getSaveFileName(self, "Save Report", default_name, f"{ext.upper()} Files (*.{ext})")
        if not path: return
        
        try:
            if fmt == "csv":
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["File", "Modality", "AI Probability", "Verdict", "Key Reasons"])
                    writer.writeheader(); writer.writerows(self.results_data)
            elif fmt == "html":
                html = "<html><head><style>body{font-family:Arial;background:#f4f4f4;} table{width:100%;border-collapse:collapse;background:white;} th{background:#333;color:white;padding:10px;} td{padding:10px;border-bottom:1px solid #ddd;} .fake{color:#d9534f;font-weight:bold;} .real{color:#5cb85c;font-weight:bold;}</style></head><body>"
                html += "<h2>DeepGuard Batch Report</h2><table><tr><th>File</th><th>Modality</th><th>AI Prob</th><th>Verdict</th><th>Reasons</th></tr>"
                for r in self.results_data:
                    cls = "fake" if "AI" in r.get("Verdict", "") else "real"
                    html += f"<tr><td>{r.get('File','')}</td><td>{r.get('Modality','').upper()}</td><td>{r.get('AI Probability','')}</td><td class='{cls}'>{r.get('Verdict','')}</td><td>{r.get('Key Reasons','')}</td></tr>"
                html += "</table></body></html>"
                with open(path, "w", encoding="utf-8") as f: f.write(html)
            elif fmt == "jsonl":
                with open(path, "w", encoding="utf-8") as f:
                    for row in self.results_data:
                        f.write(json.dumps(row) + "\n")
            QMessageBox.information(self, "Export Success", f"Report saved to:\n{path}")
        except Exception as e: QMessageBox.critical(self, "Export Error", str(e))

    def clear_table(self): self.table.setRowCount(0); self.progress.setValue(0); self.results_data = []

class BatchWorker(QThread):
    progress = Signal(int, str); row_ready = Signal(dict); finished_batch = Signal()
    def __init__(self, files): super().__init__(); self.files = files
    def run(self):
        total = len(self.files); completed = 0
        
        def analyze_file(fpath, modality):
            try:
                if modality == "audio": res = detect_audio(fpath, lambda p, m: None, False, api_mode=True)
                elif modality == "image": res = detect_image(fpath, lambda p, m: None, False, api_mode=True)
                elif modality == "video": res = detect_video(fpath, lambda p, m: None, False, api_mode=True)
                elif modality == "text": res = detect_text(fpath, lambda p, m: None, False, "", "", api_mode=True)
                score = res.get("score", 0)
                return {"File": os.path.basename(fpath), "Modality": modality, "AI Probability": f"{score*100:.1f}%", "Verdict": "AI-Generated" if score > 0.5 else "Authentic", "Key Reasons": "; ".join(res.get("reasons", []))}
            except Exception as e:
                return {"File": os.path.basename(fpath), "Modality": modality, "AI Probability": "Error", "Verdict": "Error", "Key Reasons": str(e)}

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(analyze_file, fpath, mod): (fpath, mod) for fpath, mod in self.files}
            for future in as_completed(futures):
                res = future.result()
                self.row_ready.emit(res)
                completed += 1
                self.progress.emit(int((completed / total) * 100), f"Analyzed {completed}/{total}")
                
        self.progress.emit(100, "Done"); self.finished_batch.emit()


# ===========================================================================
#  DASHBOARD TAB (With Clear Stats)
# ===========================================================================
class DashboardTab(QWidget):
    def __init__(self):
        super().__init__(); self.total_scans, self.fake_scans = 0,0; self.build_ui()

    def build_ui(self):
        outer = QVBoxLayout(self); outer.setSpacing(20)
        outer.addWidget(QLabel("📊  GLOBAL DASHBOARD", objectName="title"))
        outer.addWidget(QLabel("Session statistics and live detection feed across all modalities.", objectName="subtitle"))

        main_splitter = QSplitter(Qt.Horizontal)
        left_widget = QWidget(); lv = QVBoxLayout(left_widget)
        cards_layout = QHBoxLayout()
        self.card_total = self.make_stat_card("TOTAL SCANS", "0", "#58a6ff")
        self.card_fake = self.make_stat_card("AI DETECTED", "0", "#f85149")
        self.card_real = self.make_stat_card("LIKELY AUTHENTIC", "0", "#3fb950")
        cards_layout.addWidget(self.card_total); cards_layout.addWidget(self.card_fake); cards_layout.addWidget(self.card_real)
        lv.addLayout(cards_layout)

        feed_layout = QVBoxLayout()
        feed_header = QHBoxLayout()
        feed_header.addWidget(QLabel("Live Detection Feed", styleSheet="color:#8b949e; font-size:14px; font-weight:700;"))
        feed_header.addStretch()
        self.btn_clear_stats = QPushButton("Clear Stats", objectName="secondary")
        self.btn_clear_stats.clicked.connect(self.clear_stats)
        feed_header.addWidget(self.btn_clear_stats)
        feed_layout.addLayout(feed_header)

        self.search_input = QLineEdit(); self.search_input.setPlaceholderText("Filter feed by filename or verdict...")
        self.search_input.textChanged.connect(self.filter_table)
        feed_layout.addWidget(self.search_input)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Timestamp", "Modality", "File Name", "Verdict"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers); feed_layout.addWidget(self.table, 1)
        lv.addLayout(feed_layout)
        main_splitter.addWidget(left_widget)

        right_widget = QWidget(); rv = QVBoxLayout(right_widget)
        rv.addWidget(QLabel("Detection Ratio", styleSheet="color:#8b949e; font-size:14px; font-weight:700;"))
        if HAS_MPL:
            self.fig = Figure(figsize=(5, 4), facecolor="#0d1117" if QApplication.instance().property("Theme") == "dark" else "#ffffff")
            self.ax = self.fig.add_subplot(111)
            self.update_pie_chart()
            self.canvas = FigureCanvas(self.fig); self.canvas.setStyleSheet("background:#0d1117;")
            rv.addWidget(self.canvas, 1)
        main_splitter.addWidget(right_widget)
        main_splitter.setStretchFactor(0, 2); main_splitter.setStretchFactor(1, 1)
        outer.addWidget(main_splitter, 1)

    def make_stat_card(self, title, value, color):
        card = QFrame(objectName="card"); card.setStyleSheet(QApplication.instance().property("QSS")); layout = QVBoxLayout(card)
        lbl_title = QLabel(title); lbl_title.setStyleSheet(f"color:{color}; font-size:12px; font-weight:700;"); lbl_title.setAlignment(Qt.AlignCenter)
        lbl_val = QLabel(value, objectName="metric"); lbl_val.setStyleSheet(f"color:{color}; font-size:36px; font-weight:800;"); lbl_val.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl_title); layout.addWidget(lbl_val); return card

    def update_pie_chart(self):
        if not HAS_MPL: return
        self.ax.clear()
        real = self.total_scans - self.fake_scans
        if self.total_scans == 0:
            self.ax.text(0.5, 0.5, "No Data", ha="center", va="center", color="#8b949e", fontsize=14, transform=self.ax.transAxes)
            self.ax.set_axis_off()
        else:
            self.ax.pie([real, self.fake_scans], labels=['Authentic', 'AI-Generated'], colors=['#3fb950', '#f85149'], autopct='%1.1f%%', startangle=90, textprops={'color':'#c9d1d9' if QApplication.instance().property("Theme") == "dark" else '#24292f'})
            self.ax.set_facecolor("#0d1117" if QApplication.instance().property("Theme") == "dark" else "#ffffff")
        self.canvas.draw()

    def clear_stats(self):
        self.total_scans = 0; self.fake_scans = 0
        self.card_total.layout().itemAt(1).widget().setText("0")
        self.card_fake.layout().itemAt(1).widget().setText("0")
        self.card_real.layout().itemAt(1).widget().setText("0")
        self.table.setRowCount(0)
        self.update_pie_chart()

    def filter_table(self, text):
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 2) # Check File Name column
            if item:
                self.table.setRowHidden(row, text.lower() not in item.text().lower())

    def update_stats(self, res):
        self.total_scans += 1
        self.card_total.layout().itemAt(1).widget().setText(str(self.total_scans))
        is_fake = "AI-Generated" in res.get("verdict", "")
        if is_fake:
            self.fake_scans += 1; self.card_fake.layout().itemAt(1).widget().setText(str(self.fake_scans))
        else: self.card_real.layout().itemAt(1).widget().setText(str(self.total_scans - self.fake_scans))
        
        row = self.table.rowCount(); self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(datetime.now().strftime("%H:%M:%S")))
        self.table.setItem(row, 1, QTableWidgetItem(res.get("modality", "N/A").upper()))
        self.table.setItem(row, 2, QTableWidgetItem(res.get("file_name", "N/A")))
        verdict_item = QTableWidgetItem(res.get("verdict", "N/A"))
        verdict_item.setForeground(QColor("#f85149") if is_fake else QColor("#3fb950"))
        self.table.setItem(row, 3, verdict_item)
        self.update_pie_chart()


# ===========================================================================
#  HISTORY TAB (With Live Search)
# ===========================================================================
class HistoryTab(QWidget):
    restore_requested = Signal(str, dict)
    def __init__(self):
        super().__init__(); self.history_data = []; self.build_ui()

    def build_ui(self):
        outer = QVBoxLayout(self); outer.setSpacing(12)
        outer.addWidget(QLabel("🕒  SESSION HISTORY", objectName="title"))
        outer.addWidget(QLabel("Double-click any row to instantly restore the analysis in its respective tab.", objectName="subtitle"))
        
        ctrl = QHBoxLayout()
        self.btn_clear = QPushButton("🗑  Clear History", objectName="danger"); self.btn_clear.clicked.connect(self.clear_history)
        self.btn_export_sql = QPushButton("🗄  Export SQLite", objectName="secondary"); self.btn_export_sql.clicked.connect(self.export_sqlite)
        ctrl.addWidget(self.btn_clear); ctrl.addWidget(self.btn_export_sql); ctrl.addStretch()
        outer.addLayout(ctrl)

        search_box = QHBoxLayout()
        search_box.addWidget(QLabel("Filter:"))
        self.search_input = QLineEdit(); self.search_input.setPlaceholderText("Filter history by filename or verdict...")
        self.search_input.textChanged.connect(self.filter_table)
        search_box.addWidget(self.search_input)
        outer.addLayout(search_box)
        
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Timestamp", "Modality", "File Name", "Verdict", "AI Prob"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.doubleClicked.connect(self.on_double_click); outer.addWidget(self.table, 1)

    def filter_table(self, text):
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 2)
            if item:
                self.table.setRowHidden(row, text.lower() not in item.text().lower())

    def clear_history(self):
        self.table.setRowCount(0); self.history_data = []

    def add_entry(self, res):
        row = self.table.rowCount(); self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(datetime.now().strftime("%H:%M:%S")))
        self.table.setItem(row, 1, QTableWidgetItem(res.get("modality", "N/A").upper()))
        self.table.setItem(row, 2, QTableWidgetItem(res.get("file_name", "N/A")))
        verdict_item, score_item = QTableWidgetItem(res.get("verdict", "N/A")), QTableWidgetItem(f"{res.get('score', 0)*100:.1f}%")
        if "AI-Generated" in res.get("verdict", "") or "Deepfake" in res.get("verdict", ""):
            verdict_item.setForeground(QColor("#f85149")); score_item.setForeground(QColor("#f85149"))
        else:
            verdict_item.setForeground(QColor("#3fb950")); score_item.setForeground(QColor("#3fb950"))
        self.table.setItem(row, 3, verdict_item); self.table.setItem(row, 4, score_item)
        self.history_data.append(res)

    def on_double_click(self, index):
        if index.row() < len(self.history_data):
            res = self.history_data[index.row()]
            self.restore_requested.emit(res.get("modality", ""), res)

    def export_sqlite(self):
        if not self.history_data: return QMessageBox.warning(self, "No Data", "No history to export.")
        default_name = f"deepguard_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        path, _ = QFileDialog.getSaveFileName(self, "Save SQLite Database", default_name, "SQLite DB (*.db)")
        if not path: return
        
        try:
            conn = sqlite3.connect(path)
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS detections (
                timestamp TEXT, modality TEXT, file_name TEXT, score REAL, 
                verdict TEXT, file_hash TEXT, reasons TEXT, metrics_json TEXT
            )""")
            for res in self.history_data:
                cur.execute("""INSERT INTO detections VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", (
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    res.get("modality", ""),
                    res.get("file_name", ""),
                    res.get("score", 0.0),
                    res.get("verdict", ""),
                    res.get("file_hash", ""),
                    "\n".join(res.get("reasons", [])),
                    json.dumps(res.get("metrics", {}))
                ))
            conn.commit(); conn.close()
            QMessageBox.information(self, "Export Success", f"SQLite database saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))


# ===========================================================================
#  RESEARCH TAB
# ===========================================================================
class ResearchTab(QWidget):
    def __init__(self):
        super().__init__(); self.build_ui()

    def build_ui(self):
        outer = QVBoxLayout(self); outer.setSpacing(12)
        outer.addWidget(QLabel("📚  Frontier Academic Research", objectName="title"))
        outer.addWidget(QLabel("State-of-the-art deepfake & AI-generated content detection research, organized by modality.", objectName="subtitle"))

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea{background:transparent;} QScrollBar:vertical{background:#0d1117; width:10px;} QScrollBar::handle:vertical{background:#30363d; border-radius:5px;}")
        inner = QWidget(); v = QVBoxLayout(inner); v.setSpacing(16)

        data = {
            "Audio": [
                ("AASIST (Interspeech 2022)", "Integrated spectro-temporal graph attention. SOTA on ASVspoof 2021 LA (EER 0.83%)."),
                ("LPC Residual Energy Variance (Acoustic)", "Real human vocal tracts produce chaotic, high-frequency residual noise during unvoiced consonants (like 's', 't', 'f'). Neural TTS models often smooth out this residual energy."),
                ("Cepstral Peak Prominence (CPP) Variance (Acoustic)", "Real human speech dynamically shifts CPP between voiced consonants and unvoiced fricatives. Neural TTS maintains a mathematically rigid CPP.")
            ],
            "Image": [
                ("DIRE (CVPR 2024)", "Diffusion Reconstruction Error. Reconstructs images via pretrained diffusion model; real images yield small errors."),
                ("NPR (CVPR 2024)", "Nonlinear Probability via Re-blurring. Surpasses prior SOTA on GenImage benchmark."),
                ("Secondary JPEG Ghost Map (Q40) (Forensic)", "Re-compressing images at Q90 vs Q40 reveals 'ghost' artifacts. True images degrade uniformly; spliced regions often leave distinct ghosts at high compression levels.")
            ],
            "Video": [
                ("FTCN (ICCV 2023)", "Fine-tuned CLIP + temporal consistency. 96.7% AUC on FaceForensics++ cross-dataset."),
                ("Background Temporal Flicker (Forensic)", "Deepfakes often introduce global high-frequency noise that flickers at a different rate than the face. Tracking background pixel variance over time exposes this."),
                ("Heart Rate Variability (HRV) Proxy (Biology)", "Real human heartbeats have natural, chaotic micro-fluctuations in the time between beats (HRV). Deepfakes that synthesize an rPPG signal often make it too perfectly periodic.")
            ],
            "Text": [
                ("DetectGPT (ICML 2023)", "Perturbs input and measures log-probability drop. Zero-shot detection."),
                ("Binoculars (2024)", "Perplexity ratio between two LLMs. AUROC >0.95 across generators."),
                ("Pronoun Density Skew (Linguistics)", "LLMs are heavily RLHF-tuned for inclusivity, often resulting in an unnaturally high usage of gender-neutral pronouns ('they/them') compared to standard human writing.")
            ]
        }

        for cat, papers in data.items():
            gb = QGroupBox(f"Frontier Research: {cat}"); gl = QVBoxLayout(gb)
            for p_title, p_desc in papers:
                card = QFrame(objectName="card"); card.setStyleSheet(QApplication.instance().property("QSS"))
                cv = QVBoxLayout(card); cv.setContentsMargins(12, 10, 12, 10)
                pt = QLabel(f"📄  {p_title}"); pt.setStyleSheet("color:#58a6ff; font-weight:700; font-size:12px;"); pt.setWordWrap(True)
                ps = QLabel(p_desc); ps.setWordWrap(True); ps.setStyleSheet("color:#c9d1d9; font-size:11px;")
                cv.addWidget(pt); cv.addWidget(ps); gl.addWidget(card)
            v.addWidget(gb)

        v.addStretch(); scroll.setWidget(inner); outer.addWidget(scroll, 1)


# ===========================================================================
#  MAIN WINDOW
# ===========================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepGuard v21.0 — Advanced Multi-Modal AI-Content Detection")
        self.resize(1360, 880)
        self.config = load_config()
        self.apply_theme(self.config.get("theme", "dark"))
        self.apply_custom_rules(self.config.get("custom_rules", ""))

        self.tray_icon = QSystemTrayIcon(self)
        self.tray_icon.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        self.tray_icon.setToolTip("DeepGuard AI Detection")
        self.tray_icon.show()

        header = QWidget(); hv = QHBoxLayout(header); hv.setContentsMargins(16, 12, 16, 4)
        brand = QLabel("🛡️  DeepGuard"); brand.setStyleSheet("color:#f0f6fc; font-size:22px; font-weight:800;" if self.config.get("theme") == "dark" else "color:#24292f; font-size:22px; font-weight:800;")
        tag = QLabel("LPC Resid  •  Ghost Q40  •  BG Flicker  •  Pronoun Skew  •  XML Export  •  Live Filter")
        tag.setStyleSheet("color:#8b949e; font-size:12px;" if self.config.get("theme") == "dark" else "color:#57606a; font-size:12px;")
        hv.addWidget(brand); hv.addWidget(tag); hv.addStretch()
        
        self.btn_api = QPushButton("Start API Server", objectName="secondary")
        self.btn_api.setCheckable(True); self.btn_api.clicked.connect(self.toggle_api_server)
        hv.addWidget(self.btn_api)
        
        self.status_pill = QLabel("●  Ready")
        self.status_pill.setStyleSheet("color:#3fb950; font-size:11px; padding:4px 10px; background:#0f2a18; border:1px solid #3fb950; border-radius:10px;" if self.config.get("theme") == "dark" else "color:#1a7f37; font-size:11px; padding:4px 10px; background:#dafbe1; border:1px solid #1a7f37; border-radius:10px;")
        hv.addWidget(self.status_pill)
        self.setMenuWidget(header)

        tabs = QTabWidget()
        self.dashboard_tab = DashboardTab()
        self.audio_tab = DetectionTab("audio", "Audio File", "Audio Files (*.wav);;All Files (*)", self.config.get("sensitivity_audio", 50))
        self.image_tab = DetectionTab("image", "Image File", "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)", self.config.get("sensitivity_image", 50))
        self.video_tab = DetectionTab("video", "Video File", "Videos (*.mp4 *.avi *.mov *.mkv);;All Files (*)", self.config.get("sensitivity_video", 50))
        self.text_tab = TextDetectionTab()
        self.text_tab.sens_slider.setValue(self.config.get("sensitivity_text", 50))
        self.comparison_tab = ComparisonTab(50)
        self.history_tab = HistoryTab()
        self.webcam_tab = WebcamTab()
        self.settings_tab = SettingsTab(self.config)
        
        self.text_tab.lexicon_input.setText(self.config.get("custom_lexicon", ""))
        
        self.audio_tab.result_logged.connect(self.history_tab.add_entry)
        self.image_tab.result_logged.connect(self.history_tab.add_entry)
        self.video_tab.result_logged.connect(self.history_tab.add_entry)
        self.text_tab.result_logged.connect(self.history_tab.add_entry)
        
        self.audio_tab.result_logged.connect(self.dashboard_tab.update_stats)
        self.image_tab.result_logged.connect(self.dashboard_tab.update_stats)
        self.video_tab.result_logged.connect(self.dashboard_tab.update_stats)
        self.text_tab.result_logged.connect(self.dashboard_tab.update_stats)
        
        self.history_tab.restore_requested.connect(self.restore_tab)
        self.settings_tab.settings_changed.connect(self.apply_settings)

        tabs.addTab(self.dashboard_tab, "📊  Dashboard")
        tabs.addTab(self.webcam_tab, "🎥  Live Webcam")
        tabs.addTab(self.audio_tab, "🔊  Audio")
        tabs.addTab(self.image_tab, "🖼️  Image")
        tabs.addTab(self.video_tab, "🎬  Video")
        tabs.addTab(self.text_tab, "📝  Text")
        tabs.addTab(self.comparison_tab, "⚖️  Compare")
        self.batch_tab = BatchTab()
        tabs.addTab(self.batch_tab, "📂  Batch")
        tabs.addTab(self.history_tab, "🕒  History")
        tabs.addTab(ResearchTab(), "📚  Research")
        tabs.addTab(self.settings_tab, "⚙️  Settings")
        self.setCentralWidget(tabs)

        foot = QLabel("DeepGuard v21.0  •  LPC Residual  •  Q40 Ghost Map  •  Background Flicker  •  Pronoun Skew  •  For research use only.")
        foot.setAlignment(Qt.AlignCenter); foot.setStyleSheet("color:#6e7681; font-size:10px; padding:6px; background:#0d1117;" if self.config.get("theme") == "dark" else "color:#57606a; font-size:10px; padding:6px; background:#f6f8fa;")
        sb = self.statusBar(); sb.addWidget(foot, 1)
        
        self.sys_label = QLabel("CPU: N/A  |  RAM: N/A")
        self.sys_label.setStyleSheet("color:#8b949e; padding: 0 10px;" if self.config.get("theme") == "dark" else "color:#57606a; padding: 0 10px;")
        sb.addPermanentWidget(self.sys_label)
        
        if HAS_PSUTIL:
            self.sys_timer = QTimer(self)
            self.sys_timer.timeout.connect(self.update_sys_stats)
            self.sys_timer.start(2000)

        missing = []
        if not HAS_PIL: missing.append("Pillow")
        if not HAS_CV2: missing.append("opencv-python")
        if not HAS_SCIPY: missing.append("scipy")
        if not HAS_MPL: missing.append("matplotlib")
        if not HAS_PSUTIL: missing.append("psutil")
        if missing: self.statusBar().showMessage(f"⚠  Missing optional deps (some tabs limited): {', '.join(missing)}", 5000)
        
        self.api_thread = None

    def update_sys_stats(self):
        try:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            self.sys_label.setText(f"CPU: {cpu:.1f}%  |  RAM: {ram:.1f}%")
        except: pass

    def apply_theme(self, theme):
        qss = DARK_QSS if theme == "dark" else LIGHT_QSS
        self.setStyleSheet(qss)
        QApplication.instance().setProperty("Theme", theme)
        QApplication.instance().setProperty("QSS", qss)

    def apply_custom_rules(self, rules_str):
        QApplication.instance().setProperty("CustomRules", rules_str)

    def apply_settings(self, config):
        self.apply_theme(config.get("theme", "dark"))
        self.apply_custom_rules(config.get("custom_rules", ""))
        self.audio_tab.sens_slider.setValue(config["sensitivity_audio"])
        self.image_tab.sens_slider.setValue(config["sensitivity_image"])
        self.video_tab.sens_slider.setValue(config["sensitivity_video"])
        self.text_tab.sens_slider.setValue(config["sensitivity_text"])
        self.text_tab.lexicon_input.setText(config["custom_lexicon"])
        self.config = config
        brand = self.menuWidget().layout().itemAt(0).widget()
        tag = self.menuWidget().layout().itemAt(1).widget()
        brand.setStyleSheet("color:#f0f6fc; font-size:22px; font-weight:800;" if self.config.get("theme") == "dark" else "color:#24292f; font-size:22px; font-weight:800;")
        tag.setStyleSheet("color:#8b949e; font-size:12px;" if self.config.get("theme") == "dark" else "color:#57606a; font-size:12px;")
        self.status_pill.setStyleSheet("color:#3fb950; font-size:11px; padding:4px 10px; background:#0f2a18; border:1px solid #3fb950; border-radius:10px;" if self.config.get("theme") == "dark" else "color:#1a7f37; font-size:11px; padding:4px 10px; background:#dafbe1; border:1px solid #1a7f37; border-radius:10px;")
        foot = self.statusBar().layout().itemAt(0).widget()
        foot.setStyleSheet("color:#6e7681; font-size:10px; padding:6px; background:#0d1117;" if self.config.get("theme") == "dark" else "color:#57606a; font-size:10px; padding:6px; background:#f6f8fa;")
        self.sys_label.setStyleSheet("color:#8b949e; padding: 0 10px;" if self.config.get("theme") == "dark" else "color:#57606a; padding: 0 10px;")

    def toggle_api_server(self):
        if self.btn_api.isChecked():
            port = self.config.get("api_port", 8765)
            self.api_thread = APIServerThread(port=port)
            self.api_thread.status.connect(lambda msg: self.statusBar().showMessage(msg, 5000))
            self.api_thread.start()
            self.btn_api.setText("Stop API Server")
            self.btn_api.setStyleSheet("background: #da3633; color: white; border: none; padding: 9px 18px; border-radius: 6px; font-weight: 600;" if self.config.get("theme") == "dark" else "background: #cf222e; color: white; border: none; padding: 9px 18px; border-radius: 6px; font-weight: 600;")
        else:
            if self.api_thread:
                self.api_thread.stop(); self.api_thread = None
            self.btn_api.setText("Start API Server")
            self.btn_api.setStyleSheet("")
            self.btn_api.setObjectName("secondary")
            self.style().polish(self.btn_api)
            self.statusBar().showMessage("API Server stopped.", 3000)

    def closeEvent(self, event):
        self.config["sensitivity_audio"] = self.audio_tab.sensitivity
        self.config["sensitivity_image"] = self.image_tab.sensitivity
        self.config["sensitivity_video"] = self.video_tab.sensitivity
        self.config["sensitivity_text"] = self.text_tab.sensitivity
        self.config["custom_lexicon"] = self.text_tab.lexicon_input.text()
        save_config(self.config)
        if self.api_thread: self.api_thread.stop()
        event.accept()

    def restore_tab(self, modality, res):
        if modality == "audio": self.audio_tab.load_from_history(res)
        elif modality == "image": self.image_tab.load_from_history(res)
        elif modality == "video": self.video_tab.load_from_history(res)
        elif modality == "text": self.text_tab.load_from_history(res)

def main():
    app = QApplication(sys.argv); app.setStyle("Fusion")
    config = load_config()
    app.setProperty("Theme", config.get("theme", "dark"))
    app.setProperty("QSS", DARK_QSS if config.get("theme") == "dark" else LIGHT_QSS)
    app.setProperty("CustomRules", config.get("custom_rules", ""))
    w = MainWindow(); w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()