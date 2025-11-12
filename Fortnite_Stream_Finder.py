# fortnite_stream_finder.py
"""
Detector de streamers en tu partida de Fortnite.
Funcionalidad: captura región seleccionada, realiza OCR sobre la lista, baja con scroll suave,
compara y consulta Twitch (Helix) para marcar quién está en directo y permite abrir su canal.

Advanced Multi-Threaded Version (Optimized for Professional Fluidity):
- Implements 'Synchronized Smooth Scroll' to prevent OS I/O blocking and UI freezing.
- Uses Mean Squared Error (MSE) on **BINARIZED** images for robust end-of-list detection,
  making it **agnostic to dynamic backgrounds**.
- Strictly enforces two phases: 1) Full Capture, 2) Full Analysis.
"""

import threading
import time
import os
import requests
import webbrowser
import sys
from collections import OrderedDict
import re
import math
import queue
import datetime

import pyautogui
from PIL import Image, ImageOps
import pytesseract
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

# ---------------- CONFIG ----------------
# Ajusta la ruta si tesseract no está en PATH
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

DEFAULT_SCROLL_LINES = 300  # Número de "clicks" a bajar por captura. Se ejecutan suavemente.
DEFAULT_SCROLL_DELAY = 0.01  # Tiempo total (en segundos) que debe durar el scroll suave entre capturas.
MAX_CAPTURE_LIMIT = 500  # Límite de seguridad para capturas totales
MSE_THRESHOLD = 5.0  # <-- ¡CLAVE! Umbral de error MSE. Aumentado para tolerar ruido de fondo tras binarización.
CROP_HEIGHT_RATIO = 0.15  # Porcentaje de la parte inferior de la imagen a comparar (15%). Ignora el fondo dinámico superior.
CONSECUTIVE_SCROLL_STOP = 5  # Número de capturas idénticas (en el crop inferior binarizado) para DETENER la captura. (Rápido)
OCR_LANG = 'eng'  # idioma para Tesseract


# ----------------------------------------

# ---------- Twitch helpers (omitted for brevity) ----------
def get_twitch_app_token(client_id, client_secret):
    url = "https://id.twitch.tv/oauth2/token"
    params = {"client_id": client_id, "client_secret": client_secret, "grant_type": "client_credentials"}
    r = requests.post(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()["access_token"]


def get_streams_for_logins(logins, client_id, token):
    if not logins:
        return []
    url = "https://api.twitch.tv/helix/streams"
    headers = {"Client-ID": client_id, "Authorization": f"Bearer {token}"}
    streams = []
    for i in range(0, len(logins), 100):
        batch = logins[i:i + 100]
        params = []
        for n in batch:
            params.append(("user_login", n))
        r = requests.get(url, headers=headers, params=params, timeout=12)
        r.raise_for_status()
        streams.extend(r.json().get("data", []))
    return streams


def search_twitch_channel_by_name(name, client_id, token):
    url = "https://api.twitch.tv/helix/search/channels"
    headers = {"Client-ID": client_id, "Authorization": f"Bearer {token}"}
    params = {"query": name, "first": 10}
    r = requests.get(url, headers=headers, params=params, timeout=8)
    r.raise_for_status()
    return r.json().get("data", [])


# ---------- OCR & image processing ----------

def _preprocess_for_comparison(pil_img):
    """
    Faster preprocessing to generate a high-contrast binary array for MSE comparison.
    This step effectively isolates the list structure (text, lines) from dynamic backgrounds.
    """
    gray = ImageOps.grayscale(pil_img)
    arr = np.array(gray)

    # Aggressive Thresholding: isolate list text/structure
    # Use Otsu's method for automatic optimal threshold
    _, thresh = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return thresh.astype("float")  # Return as float for MSE calculation


def preprocess_image_for_ocr(pil_img):
    w, h = pil_img.size
    scale = 2.0 if w < 900 else 1.0
    if scale != 1.0:
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    gray = ImageOps.grayscale(pil_img)
    arr = np.array(gray)

    arr = cv2.bilateralFilter(arr, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr = clahe.apply(arr)
    kernel = np.array([[0, -1, 0], [-1, 6, -1], [0, -1, 0]])
    arr = cv2.filter2D(arr, -1, kernel)

    arr = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 11, 2)

    return Image.fromarray(arr)


def clean_ocr_name(name):
    if not name:
        return None
    s = name.strip()
    s = ''.join(ch for ch in s if 32 <= ord(ch) <= 126)
    s = s.replace(' ', '').replace('-', '').replace('.', '').replace('|', 'l')
    cleaned = re.sub(r'[^A-Za-z0-9_]', '', s)
    if not cleaned:
        return None
    return cleaned


def extract_nicknames_from_image(pil_img):
    proc = preprocess_image_for_ocr(pil_img)
    whitelist = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234s6789_ '
    config = f'--oem 3 --psm 6 -c tessedit_char_whitelist={whitelist}'
    try:
        text = pytesseract.image_to_string(proc, lang=OCR_LANG, config=config)
    except Exception:
        text = pytesseract.image_to_string(proc, lang=OCR_LANG)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    candidates = []
    for l in lines:
        if 2 <= len(l) <= 60:
            cleaned = clean_ocr_name(l)
            if cleaned:
                candidates.append(cleaned)
    return candidates


# ---------- UI utilities (omitted for brevity) ----------
class Tooltip:

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)
        widget.bind("<ButtonPress>", self.hide)

    def show(self, _=None):
        if self.tip:
            return
        wx = self.widget.winfo_rootx()
        wy = self.widget.winfo_rooty()
        ww = self.widget.winfo_width()
        wh = self.widget.winfo_height()
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.attributes("-topmost", True)
        label = tk.Label(self.tip, text=self.text, justify='left', background="#071215", foreground="#9cff7f",
                         relief='solid', borderwidth=1, font=("Consolas", 9), padx=6, pady=3)
        label.pack()
        self.tip.update_idletasks()
        tw = self.tip.winfo_width()
        th = self.tip.winfo_height()
        x = wx + max(0, ww // 2 - tw // 2)
        y_above = wy - th - 8
        y_below = wy + wh + 8
        if y_above > 20:
            y = y_above
        else:
            y = y_below
        screen_w = self.widget.winfo_screenwidth()
        if x + tw > screen_w - 10:
            x = screen_w - tw - 10
        if x < 10:
            x = 10
        self.tip.wm_geometry(f"+{x}+{y}")

    def hide(self, _=None):
        if self.tip:
            try:
                self.tip.destroy()
            except Exception:
                pass
            self.tip = None


# ---------- GUI & main logic ----------
class StreamFinderApp:
    def __init__(self, root):
        self.root = root
        pyautogui.PAUSE = 0.0

        root.title("Fortnite Stream Finder")
        root.attributes("-topmost", True)
        root.resizable(False, False)

        style = ttk.Style()
        try:
            style.theme_use('clam')
        except Exception:
            pass
        style.configure("TFrame", background="#071215")
        style.configure("TLabel", background="#071215", foreground="#c7f9b6", font=("Consolas", 10))
        style.configure("Header.TLabel", font=("Consolas", 13, "bold"), foreground="#9cff7f")
        style.configure("Accent.TButton", background="#07221a", foreground="#dfffd8", font=("Consolas", 10))
        style.map("Accent.TButton", background=[('active', '#13684f')])

        self.region = None
        self.scroll_step_lines = tk.IntVar(value=DEFAULT_SCROLL_LINES)
        self.scroll_delay = tk.DoubleVar(value=DEFAULT_SCROLL_DELAY)
        self.status_text = tk.StringVar(value="Selecciona región y pulsa Start scan.")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.twitch_client_id = tk.StringVar(value="")
        self.twitch_client_secret = tk.StringVar(value="")
        self.results = OrderedDict()
        self.results_lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.total_images_produced = 0

        frm = ttk.Frame(root, padding=14, style="TFrame")
        frm.grid(column=0, row=0, sticky="nsew")

        header = ttk.Label(frm, text="▶ Fortnite Stream Finder", style="Header.TLabel")
        header.grid(column=0, row=0, columnspan=3, sticky="w", pady=(0, 12))

        btn_region = ttk.Button(frm, text="⤓ Select region", style="Accent.TButton", command=self.select_region)
        btn_region.grid(column=0, row=1, sticky="w", pady=(2, 12), ipadx=6, ipady=4)
        Tooltip(btn_region, "Select on screen by dragging a rectangle over the list of players.")

        cfg_frame = ttk.Frame(frm, style="TFrame")
        cfg_frame.grid(column=0, row=2, sticky="w", pady=(0, 10))

        ttk.Label(cfg_frame, text="Scroll lines:", style="TLabel").grid(column=0, row=0, sticky="w", padx=(0, 4))
        e_scroll_lines = ttk.Entry(cfg_frame, textvariable=self.scroll_step_lines, width=6)
        e_scroll_lines.grid(column=1, row=0, sticky="w", padx=(0, 12))
        Tooltip(e_scroll_lines, "Número de 'clicks' de scroll. Se ejecutan suavemente en varios pasos.")

        ttk.Label(cfg_frame, text="Delay (s):", style="TLabel").grid(column=2, row=0, sticky="w", padx=(0, 4))
        e_delay = ttk.Entry(cfg_frame, textvariable=self.scroll_delay, width=6)
        e_delay.grid(column=3, row=0, sticky="w", padx=(0, 0))
        Tooltip(e_delay, "Tiempo total para el scroll entre capturas. Ajustado a 0.05s por defecto.")

        ttk.Separator(frm, orient=tk.HORIZONTAL).grid(column=0, row=3, sticky="ew", pady=10, columnspan=3)
        ttk.Label(frm, text="Twitch CLIENT_ID:", style="TLabel").grid(column=0, row=4, sticky="w")
        ttk.Entry(frm, textvariable=self.twitch_client_id, width=44).grid(column=0, row=5, sticky="w", pady=(4, 8))
        Tooltip(frm.nametowidget(frm.winfo_children()[-1]),
                "CLIENT_ID de tu aplicación Twitch (requerido para verificación en vivo).")
        ttk.Label(frm, text="Twitch CLIENT_SECRET:", style="TLabel").grid(column=0, row=6, sticky="w")
        ttk.Entry(frm, textvariable=self.twitch_client_secret, show="*", width=44).grid(column=0, row=7, sticky="w",
                                                                                        pady=(4, 8))

        self.btn_start = ttk.Button(frm, text="▶ Start scan", style="Accent.TButton", command=self.start_scan)
        self.btn_start.grid(column=0, row=8, sticky="ew", pady=(8, 6))
        Tooltip(self.btn_start, "Iniciar escaneo: Captura (rápida y fluida), luego OCR y verificación.")

        self.btn_stop = ttk.Button(frm, text="■ Stop", style="Accent.TButton", command=self.request_stop,
                                   state="disabled")
        self.btn_stop.grid(column=0, row=9, sticky="ew", pady=(0, 10))
        Tooltip(self.btn_stop, "Detener escaneo (F4). Siempre responde.")

        root.bind("<F4>", lambda e: self.request_stop())

        btn_manual = ttk.Button(frm, text="🔎 Search streamer manually", command=self.search_manual_streamer)
        btn_manual.grid(column=0, row=10, sticky="ew", pady=(4, 12))
        Tooltip(btn_manual, "Buscar nombre manualmente en Twitch (requiere credenciales).")

        progress = ttk.Progressbar(frm, maximum=100, variable=self.progress_var, length=520, mode='determinate')
        progress.grid(column=0, row=11, pady=(2, 6))
        ttk.Label(frm, textvariable=self.status_text, style="TLabel").grid(column=0, row=12, sticky="w", pady=(0, 10))

        ttk.Separator(frm, orient=tk.HORIZONTAL).grid(column=0, row=13, sticky="ew", pady=10)
        ttk.Label(frm, text="Results (only LIVE button displayed):", style="TLabel").grid(column=0, row=14, sticky="w",
                                                                                          pady=(0, 8))

        container = ttk.Frame(frm)
        container.grid(column=0, row=15, sticky="nsew")
        self.canvas = tk.Canvas(container, height=300, bg="#02060a", highlightthickness=0)
        self.vscroll = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vscroll.set)
        self.results_inner = ttk.Frame(self.canvas, style="TFrame")
        self.results_inner_id = self.canvas.create_window((0, 0), window=self.results_inner, anchor='nw')

        def _on_frame_config(e):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        def _on_canvas_config(event):
            self.canvas.itemconfig(self.results_inner_id, width=event.width)

        self.results_inner.bind("<Configure>", _on_frame_config)
        self.canvas.bind("<Configure>", _on_canvas_config)

        def _bind_wheel(ev):
            self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
            self.canvas.bind_all("<Button-4>", _on_mousewheel)
            self.canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_wheel(ev):
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")

        def _on_mousewheel(event):
            if event.delta:
                delta = -1 * int(event.delta / 120)
            else:
                delta = 1 if event.num == 5 else -1
            self.canvas.yview_scroll(delta, "units")

        self.canvas.bind("<Enter>", _bind_wheel)
        self.canvas.bind("<Leave>", _unbind_wheel)

        self.canvas.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self.vscroll.pack(side="right", fill="y")

        self.root.configure(bg="#02060a")

    # ---------- region selection (omitted for brevity) ----------
    def select_region(self):
        messagebox.showinfo("Select region",
                            "Drag over the player list to select the region. Tap and release to confirm.")
        overlay = tk.Toplevel(self.root)
        overlay.attributes("-fullscreen", True)
        overlay.attributes("-alpha", 0.25)
        overlay.attributes("-topmost", True)
        overlay.configure(bg="#000000")
        canvas = tk.Canvas(overlay, cursor="cross", bg="#111111")
        canvas.pack(fill=tk.BOTH, expand=True)

        start_x = start_y = end_x = end_y = 0
        rect = None

        def on_button_press(event):
            nonlocal start_x, start_y, rect
            start_x, start_y = event.x, event.y
            rect = canvas.create_rectangle(start_x, start_y, start_x, start_y, outline="#00ff7f", width=2)

        def on_move(event):
            nonlocal rect
            if rect:
                canvas.coords(rect, start_x, start_y, event.x, event.y)

        def on_button_release(event):
            nonlocal end_x, end_y
            end_x, end_y = event.x, event.y
            x1 = overlay.winfo_rootx() + min(start_x, end_x)
            y1 = overlay.winfo_rooty() + min(start_y, end_y)
            x2 = overlay.winfo_rootx() + max(start_x, end_x)
            y2 = overlay.winfo_rooty() + max(start_y, end_y)
            overlay.destroy()
            if x2 <= x1 or y2 <= y1 or (x2 - x1) < 50 or (y2 - y1) < 20:
                messagebox.showerror("Failure", "Invalid region. Please select a larger area again.")
                return
            self.region = (x1, y1, x2, y2)
            self._set_status(f"Saved region: {self.region}")

        canvas.bind("<ButtonPress-1>", on_button_press)
        canvas.bind("<B1-Motion>", on_move)
        canvas.bind("<ButtonRelease-1>", on_button_release)
        overlay.mainloop()

    def request_stop(self):
        if not self.stop_flag.is_set():
            self.stop_flag.set()
            self._set_status("Stopping scan...")
            self.root.after(0, lambda: self.btn_stop.config(text="■ Stopping...", state="disabled"))

    def start_scan(self):
        if not self.region:
            messagebox.showerror("Undefined region", "Select the region before starting the search.")
            return
        self.stop_flag.clear()
        for widget in self.results_inner.winfo_children():
            widget.destroy()

        with self.results_lock:
            self.results.clear()

        self.progress_var.set(0)
        self._set_status("Starting scan...")
        self.btn_start.config(state="disabled")
        self.btn_stop.config(text="■ Stop (F4)", state="normal")
        self.total_images_produced = 0

        # Se inicia el hilo principal que gestiona las dos fases
        t = threading.Thread(target=self._scan_manager_thread, daemon=True)
        t.start()

    # ---------- scanning logic (Producer-Consumer - OPTIMIZED) ----------

    def _scan_manager_thread(self):
        """Manages the two distinct phases: Capture (Producer) and Analysis (Consumers)."""
        images_queue = queue.Queue()  # Cola ilimitada para capturas

        # --- Phase 1: Capture Loop (Producer) ---
        self._scan_producer_thread(images_queue)

        # Check if stopped during capture
        if self.stop_flag.is_set():
            self._cleanup_after_stop()
            return

        # --- Phase 2: Analysis Wait (Consumers) ---

        token = None
        client_id = self.twitch_client_id.get().strip()
        client_secret = self.twitch_client_secret.get().strip()

        # 1. Obtener Token de Twitch (si es posible)
        if client_id and client_secret:
            try:
                self._set_status("Phase 2: Obtaining a Twitch token...")
                token = get_twitch_app_token(client_id, client_secret)
            except Exception as e:
                self._set_status(f"Phase 2: Twitch token error: {e}. Continuing without verification.")
                token = None

        # 2. Iniciar Hilos de Análisis (Consumidores)
        num_workers = os.cpu_count() or 4
        self._set_status(f"Phase 2: Starting {num_workers} analysis workers...")
        workers = []
        for _ in range(num_workers):
            t = threading.Thread(target=self._worker_thread,
                                 args=(images_queue, token, client_id),
                                 daemon=True)
            t.start()
            workers.append(t)

        # 3. Enviar Sentinel (None) para detener a los trabajadores cuando la cola esté vacía
        for _ in range(num_workers):
            images_queue.put(None)

        self.progress_var.set(10)

        # 4. Esperar a que la cola se vacíe
        while images_queue.unfinished_tasks > num_workers and not self.stop_flag.is_set():
            tasks_remaining = images_queue.unfinished_tasks - num_workers
            tasks_completed = self.total_images_produced - tasks_remaining

            # La barra de progreso de 10% a 100%
            progress_analysis = 10 + (90 * tasks_completed / self.total_images_produced)
            self.progress_var.set(progress_analysis)
            self._set_status(f"Phase 2: Analyzing... {tasks_completed}/{self.total_images_produced} images analyzed.")
            time.sleep(0.5)

        images_queue.join()

        self._cleanup_after_stop()

    def _scan_producer_thread(self, images_queue):
        """
        The 'Producer' thread (Capture Phase). Uses synchronized smooth scroll and MSE detection
        on a cropped area of a Binarized image to ignore dynamic backgrounds.
        """
        x1, y1, x2, y2 = self.region
        width = x2 - x1
        height = y2 - y1

        # CROP CONFIG: Solo se compara una franja inferior para ignorar fondos dinámicos.
        crop_h = max(10, int(height * CROP_HEIGHT_RATIO))  # Asegura al menos 10px

        delay = max(0.005, float(self.scroll_delay.get()))
        scroll_lines = max(1, int(self.scroll_step_lines.get()))

        previous_cropped_arr = None  # Almacenará el array RECORTADO y BINARIZADO de la captura anterior
        max_no_new = CONSECUTIVE_SCROLL_STOP
        captures = 0
        no_new_in_row = 0
        mse_threshold = MSE_THRESHOLD

        total_scroll_steps = scroll_lines
        micro_pause = delay / total_scroll_steps if total_scroll_steps > 0 else 0.005

        # IMPORTANT: Mover el ratón al centro de la región para asegurar el enfoque del scroll
        try:
            mid_x, mid_y = x1 + width // 2, y1 + height // 2
            pyautogui.moveTo(mid_x, mid_y, duration=0.01)
        except Exception:
            pass

        while not self.stop_flag.is_set() and captures < MAX_CAPTURE_LIMIT:
            # 1. Capture and Prepare
            img = pyautogui.screenshot(region=(x1, y1, width, height))

            # **NUEVO:** Preprocesamiento de la imagen para aislar el texto/líneas
            # y hacer el MSE agnóstico al fondo dinámico.
            current_arr_comp = _preprocess_for_comparison(img)

            # 2. CROP: Recortar solo la parte inferior para la detección de fin de scroll
            # Se compara solo la franja inferior donde el contenido deja de entrar.
            current_cropped_arr = current_arr_comp[-crop_h:, :]

            if previous_cropped_arr is not None:
                # 3. Robust End-of-List Detection (Mean Squared Error - MSE)
                # Cálculo: Suma de errores al cuadrado dividido por el número total de píxeles
                error = np.sum((current_cropped_arr - previous_cropped_arr) ** 2) / (current_cropped_arr.size)

                if error < mse_threshold:
                    no_new_in_row += 1
                else:
                    no_new_in_row = 0

                # ¡DETENCIÓN CRÍTICA!
                if no_new_in_row >= max_no_new:
                    # Antes de detener, añade la última imagen válida (que es la que repitió 'max_no_new' veces)
                    images_queue.put(img)
                    self.total_images_produced += 1
                    break

                    # Guardar el array RECORTADO y BINARIZADO actual para la siguiente comparación
            previous_cropped_arr = current_cropped_arr

            # Poner la imagen COMPLETA capturada en la cola para el análisis futuro
            images_queue.put(img)
            self.total_images_produced += 1

            captures += 1

            prog = min(10, int((captures / MAX_CAPTURE_LIMIT) * 10))
            self.progress_var.set(prog)
            self.root.after(0, lambda: self._set_status(
                f"Phase 1: Capturing... {self.total_images_produced} images produced. (screen locked)"))

            # 4. SYNCHRONIZED SMOOTH SCROLL
            for _ in range(total_scroll_steps):
                try:
                    pyautogui.scroll(-1)  # Scroll 1 line down
                except Exception:
                    pass

                time.sleep(micro_pause)
                if self.stop_flag.is_set():
                    return  # Detener el productor inmediatamente

        # Fin del bucle de captura

    def _worker_thread(self, images_queue, token, client_id):
        """The 'Consumer' thread. Pulls images from the queue, performs OCR and Twitch checks."""
        while not self.stop_flag.is_set():
            try:
                # Bloqueo hasta que haya un elemento o se detenga el programa
                img = images_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if img is None:  # Sentinel received
                images_queue.task_done()
                break

            try:
                # 1. Slow CPU Task: OCR
                names = extract_nicknames_from_image(img)
                if not names:
                    continue

                # 2. Filter for new names and update internal list
                new_names_to_check = []
                with self.results_lock:
                    for n in names:
                        key_candidate = clean_ocr_name(n)
                        if not key_candidate:
                            continue
                        key = key_candidate.lower()
                        if key not in self.results:
                            self.results[key] = {'original': n, 'url': None, 'stream': None}
                            new_names_to_check.append(n)

                # 3. Slow Network Task: Twitch Check
                if new_names_to_check and token:
                    self._check_twitch_and_update(new_names_to_check, token, client_id)

            except Exception as e:
                print(f"Error in worker thread: {e}")
            finally:
                images_queue.task_done()

    def _check_twitch_and_update(self, names, token, client_id):

        def valid_twitch_login(s):
            if not s: return False
            if len(s) < 3 or len(s) > 25: return False
            return re.fullmatch(r'[a-z0-9_]+', s) is not None

        qlist_map = {}
        for nick in names:
            base_clean = clean_ocr_name(nick)
            if not base_clean: continue

            candidate = re.sub(r'[^a-z0-9_]', '', base_clean.lower())
            if valid_twitch_login(candidate):
                qlist_map[candidate] = nick

            alt = candidate.replace('_', '')
            if valid_twitch_login(alt) and alt != candidate:
                qlist_map[alt] = nick

        qlist = list(qlist_map.keys())
        if not qlist:
            return

        streams = []
        try:
            streams = get_streams_for_logins(qlist, client_id, token)
        except Exception as e:
            print(f"Twitch get_streams error: {e}")

        stream_logins = {s['user_login'].lower(): s for s in streams}

        processed_originals = set()

        with self.results_lock:
            for login, stream in stream_logins.items():
                if login in qlist_map:
                    original_name = qlist_map[login]
                    original_key = clean_ocr_name(original_name).lower()

                    self.results[original_key]['stream'] = stream
                    self.results[original_key]['url'] = f"https://twitch.tv/{stream['user_login']}"
                    self.results[original_key]['login'] = stream['user_login']
                    processed_originals.add(original_name)

            for original_name in names:
                if original_name in processed_originals:
                    continue

                original_key = clean_ocr_name(original_name).lower()
                if not self.results[original_key]['url']:
                    try:
                        hits = search_twitch_channel_by_name(original_name, client_id, token)
                        chosen = None
                        for h in hits:
                            if h.get('display_name', '').lower() == original_name.lower() or h.get('broadcaster_login',
                                                                                                   '').lower() == original_name.lower():
                                chosen = h
                                break
                        if not chosen and hits:
                            chosen = hits[0]

                        if chosen:
                            self.results[original_key]['url'] = f"https://twitch.tv/{chosen.get('broadcaster_login')}"
                            self.results[original_key]['login'] = chosen.get('broadcaster_login')
                    except Exception as e:
                        print(f"Twitch search_channel error: {e}")

        self.root.after(0, self._display_results)

    def _display_results(self):

        def build():
            for w in self.results_inner.winfo_children():
                w.destroy()

            with self.results_lock:
                items_to_display = list(self.results.items())

            row = 0
            for _, info in items_to_display:
                nick = info['original']

                bg = "#051216" if (row % 2 == 0) else "#041014"
                container = tk.Frame(self.results_inner, bg=bg)
                container.grid(column=0, row=row, sticky="ew", padx=6, pady=6)
                container.columnconfigure(0, weight=1)
                container.columnconfigure(1, weight=0)

                txt = nick
                if info['stream']:
                    login = info.get('login') or (info['stream'].get('user_login') if info['stream'] else None)
                    txt += f"  [LIVE: {login}]"

                lbl = tk.Label(container, text=txt, bg=bg, fg="#c7f9b6", font=("Consolas", 10), anchor="w")
                lbl.grid(column=0, row=0, sticky="w", padx=(8, 8), pady=6)

                if info['stream'] and info.get('url'):
                    btn = tk.Button(container, text="Go to channel", bg="#0b3b2f", fg="#dfffd8", font=("Consolas", 10),
                                    activebackground="#13684f", command=lambda u=info['url']: webbrowser.open(u))
                    btn.grid(column=1, row=0, sticky="e", padx=(12, 10), pady=6)
                    Tooltip(btn, "Go to Twich channel (only if streamming).")
                row += 1

            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        try:
            self.root.after(0, build)
        except tk.TclError:
            pass

    def _set_status(self, text):
        try:
            self.root.after(0, lambda: self.status_text.set(text))
        except tk.TclError:
            pass

    def _cleanup_after_stop(self):
        self.progress_var.set(100)
        found_live = 0
        with self.results_lock:
            found_live = sum(1 for v in self.results.values() if v['stream'])

        self._set_status(f"Scan complete. Live Streamers: {found_live}")

        self.root.after(0, lambda: (
            self.btn_start.config(state="normal"),
            self.btn_stop.config(text="■ Stop", state="disabled")
        ))

    def search_manual_streamer(self):
        if not self.twitch_client_id.get().strip() or not self.twitch_client_secret.get().strip():
            messagebox.showinfo("Twitch API required",
                                "You must enter CLIENT_ID and CLIENT_SECRET to use the manual search.")
            return
        streamer_name = simpledialog.askstring("Search streamer", "Enter streamer name:", parent=self.root)
        if not streamer_name:
            return
        try:
            token = get_twitch_app_token(self.twitch_client_id.get().strip(), self.twitch_client_secret.get().strip())
            results = search_twitch_channel_by_name(streamer_name.strip(), self.twitch_client_id.get().strip(), token)
            if not results:
                messagebox.showinfo("Result", f"The streamer could not be found {streamer_name}.")
                return
            hits_text = ""
            for r in results[:6]:
                hits_text += f"{r.get('display_name')} ({r.get('broadcaster_login')}) - LIVE: {r.get('is_live')}\n"
            messagebox.showinfo(f"Results for {streamer_name}", hits_text)
        except Exception as e:
            messagebox.showerror("Twich API error", str(e))


def main():
    root = tk.Tk()
    app = StreamFinderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()