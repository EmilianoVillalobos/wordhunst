import json
import time
import pyautogui
import cv2
import easyocr
import tensorflow as tf
import os
import numpy as np
from pynput import keyboard
from Quartz.CoreGraphics import (
    CGEventCreateMouseEvent,
    CGEventPost,
    CGEventCreate,
    CGEventGetLocation,
    CGDisplayBounds,
    CGMainDisplayID,
    kCGEventMouseMoved,
    kCGEventLeftMouseDown,
    kCGEventLeftMouseUp,
    kCGEventLeftMouseDragged,
    kCGHIDEventTap,
    kCGMouseButtonLeft,
)

ROWS = 4
COLS = 4
stop_requested = False
listener = None
CALIBRATION_FILE = "board_calibration.json"
MIN_WORD_LEN = 3
DRAG_SEGMENT_DURATION = 0.04
DRAG_SEGMENT_STEPS = 4
SLEEP_BETWEEN_LETTERS = 0.005
SLEEP_BETWEEN_WORDS = 0.02
WORD_LIST_CACHE = None
READER = easyocr.Reader(["en"], gpu=False)

IMG_SIZE = 32  # same as in train_cnn.py

MODEL_PATH = "models/wordhunt_cnn.h5"
LABELMAP_PATH = "models/label_map.json"

# made own model
CNN_MODEL = tf.keras.models.load_model(MODEL_PATH)
with open(LABELMAP_PATH, "r") as f:
    label_data = json.load(f)

IDX_TO_LABEL = {int(k): v for k, v in label_data["idx_to_label"].items()}

def predict_letter_from_tile(tile_bgr):
    """
    tile_bgr: np.ndarray (H, W, 3) in BGR (OpenCV format).
    Returns a single uppercase letter 'A'-'Z'.
    """
    gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    gray = gray.astype("float32") / 255.0

    inp = gray[np.newaxis, ..., np.newaxis]

    preds = CNN_MODEL.predict(inp, verbose=0)[0]
    idx = int(np.argmax(preds))
    letter = IDX_TO_LABEL[idx]
    return letter


NEIGHBORS = [[] for _ in range(16)]
for i in range(16):
    left = (i % 4 != 0)
    right = (i % 4 != 3)
    top = (i >= 4)
    bottom = (i < 12)
    if top and left:
        NEIGHBORS[i].append(i - 5)
    if top and right:
        NEIGHBORS[i].append(i - 3)
    if top:
        NEIGHBORS[i].append(i - 4)
    if left:
        NEIGHBORS[i].append(i - 1)
    if right:
        NEIGHBORS[i].append(i + 1)
    if bottom and left:
        NEIGHBORS[i].append(i + 3)
    if bottom:
        NEIGHBORS[i].append(i + 4)
    if bottom and right:
        NEIGHBORS[i].append(i + 5)


class WordHunt:
    def __init__(self, board: list):
        self.board = [letter.lower() for letter in board]
        self.answers = {}
        self.sorted = []
        self.word_list = {}
        self.verify_board()
        self.get_word_list()
        self.solve_board()
        self.sort_answers()

    def verify_board(self):
        if len(self.board) != 16:
            raise IndexError("Board must have exactly 16 letters")
        for letter in self.board:
            if len(letter) != 1:
                raise ValueError("Letters in board must be a single character in length")
            if not 97 <= ord(letter) <= 122:
                raise ValueError("Board must contain only letters")

    def get_word_list(self):
        global WORD_LIST_CACHE
        if WORD_LIST_CACHE is None:
            with open("wordlist.json") as file:
                WORD_LIST_CACHE = json.load(file)
        self.word_list = WORD_LIST_CACHE

    def solve_board(self):
        for cell_index in range(len(self.board)):
            cell = self.board[cell_index]
            possible_words = self.word_list.get(cell, [])
            if not possible_words:
                continue
            self.answers.update(self.solve_adjacent_cells(cell_index, [], possible_words))

    def sort_answers(self):
        self.sorted = sorted(list(self.answers.keys()), key=len, reverse=True)

    def solve_adjacent_cells(self, cell_index: int, previous_word_indexes: list, possible_words: list):
        results = {}
        current_word_indexes = previous_word_indexes.copy()
        current_word_indexes.append(cell_index)
        current_word = self.index_to_word(current_word_indexes)
        new_possible_words = []
        if len(current_word_indexes) >= 3:
            for word in possible_words:
                if word == current_word:
                    results[word] = current_word_indexes
                elif word.startswith(current_word):
                    new_possible_words.append(word)
        else:
            new_possible_words = possible_words
        if len(new_possible_words) > 0:
            used = set(current_word_indexes)
            for adjacent_cell in NEIGHBORS[cell_index]:
                if adjacent_cell in used:
                    continue
                results.update(self.solve_adjacent_cells(adjacent_cell, current_word_indexes, new_possible_words))
        return results

    def index_to_word(self, word_indexes: list):
        word = ""
        for letter_index in word_indexes:
            word += self.board[letter_index]
        return word


def on_press(key):
    global stop_requested
    try:
        if key == keyboard.Key.esc:
            stop_requested = True
            print("\n[ESC pressed] Stop requested (will stop after this word).\n")
            return False
    except Exception:
        pass


def init_stop_hotkey():
    global listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()


def get_mouse_position():
    event = CGEventCreate(None)
    loc = CGEventGetLocation(event)
    return int(loc.x), int(loc.y)


def post_mouse_event(event_type, x, y):
    event = CGEventCreateMouseEvent(None, event_type, (x, y), kCGMouseButtonLeft)
    CGEventPost(kCGHIDEventTap, event)


def move_mouse(x, y):
    post_mouse_event(kCGEventMouseMoved, x, y)


def mouse_down(x, y):
    post_mouse_event(kCGEventLeftMouseDown, x, y)


def mouse_up(x, y):
    post_mouse_event(kCGEventLeftMouseUp, x, y)


def drag_line(x1, y1, x2, y2, duration=DRAG_SEGMENT_DURATION, steps=DRAG_SEGMENT_STEPS):
    dx = (x2 - x1) / steps
    dy = (y2 - y1) / steps
    dt = duration / steps if steps > 0 else 0
    for i in range(1, steps + 1):
        nx = x1 + dx * i
        ny = y1 + dy * i
        post_mouse_event(kCGEventLeftMouseDragged, nx, ny)
        if dt > 0:
            time.sleep(dt)


def save_board_geometry(board_geometry, filename=CALIBRATION_FILE):
    if board_geometry is None:
        return
    data = {
        "top_left_x": board_geometry[0],
        "top_left_y": board_geometry[1],
        "cell_width": board_geometry[2],
        "cell_height": board_geometry[3],
    }
    try:
        with open(filename, "w") as f:
            json.dump(data, f)
        print(f"Saved board calibration to {filename}.")
    except OSError as e:
        print(f"Warning: could not save calibration to {filename}: {e}")


def load_board_geometry(filename=CALIBRATION_FILE):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        board_geometry = (
            data["top_left_x"],
            data["top_left_y"],
            data["cell_width"],
            data["cell_height"],
        )
        print(f"Loaded board calibration from {filename}.")
        return board_geometry
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        print(f"No valid saved calibration found in {filename}.")
        return None
    except OSError as e:
        print(f"Warning: could not read calibration file {filename}: {e}")
        return None
    
def save_board_image(filename=CALIBRATION_FILE):
    with open(filename, "r") as f:
        data = json.load(f)
    board_geometry = (
        data["top_left_x"],
        data["top_left_y"],
        data["cell_width"],
        data["cell_height"],
    )
    print("hi")
    width_cushion = board_geometry[2]
    height_cushion = board_geometry[3]
    screenshot_top_left_x = board_geometry[0] - width_cushion
    screenshot_top_left_y = board_geometry[1] - height_cushion

    left   = int(round(screenshot_top_left_x))
    top    = int(round(screenshot_top_left_y))
    width  = int(round(width_cushion * (COLS + 1)))
    height = int(round(height_cushion * (ROWS + 1)))
    img = pyautogui.screenshot(region=(left, top, width, height))

    img.save("board.png")
    return

def ocr_board_from_screen(cal_file="board_calibration.json"):
    """
    Uses screen coordinates of the CENTER of tile (0,0)
    plus cell width/height to accurately OCR each tile.
    Returns a 4x4 list of letters.
    """

    with open(cal_file, "r") as f:
        data = json.load(f)

    center_x = data["top_left_x"]      # actually center of tile (0,0)
    center_y = data["top_left_y"]      # actually center of tile (0,0)
    cell_w   = data["cell_width"]      # tile width in SCREEN pixels
    cell_h   = data["cell_height"]     # tile height in SCREEN pixels

    # go to top left corner of top left tile
    top_left_x = center_x - (cell_w / 2)
    top_left_y = center_y - (cell_h / 2)

    # for some reason when this is 0 i error out idk
    inner_margin_x = int(cell_w * 0.0001)
    inner_margin_y = int(cell_h * 0.0001)

    board_letters = []

    for r in range(ROWS):
        row_letters = []
        for c in range(COLS):

            # screen grab the tile
            tile_left   = top_left_x + c * cell_w
            tile_top    = top_left_y + r * cell_h
            tile_right  = tile_left + cell_w
            tile_bottom = tile_top + cell_h

            # again idk why i had to do the margin it was tweaking
            left   = int(round(tile_left   + inner_margin_x))
            top    = int(round(tile_top    + inner_margin_y))
            right  = int(round(tile_right  - inner_margin_x))
            bottom = int(round(tile_bottom - inner_margin_y))

            width  = right - left
            height = bottom - top

            tile_pil = pyautogui.screenshot(region=(left, top, width, height))
            tile = cv2.cvtColor(np.array(tile_pil), cv2.COLOR_RGB2BGR)

            # boom this shit to own model
            letter = predict_letter_from_tile(tile)

            row_letters.append(letter)

        board_letters.append(row_letters)

    return "".join(ch.lower() for row in board_letters for ch in row)

def get_mouse_position_with_prompt(msg):
    input(msg + " Then press Enter in this window...")
    x, y = get_mouse_position()
    print(f"Captured: ({x}, {y})")
    return x, y


def calibrate_board():
    print("=== Board calibration ===")
    time.sleep(0.3)
    print("1) Move your mouse to the CENTER of the TOP-LEFT cell of the grid.")
    top_left_x, top_left_y = get_mouse_position_with_prompt("   ")
    print("2) Move your mouse to the CENTER of the BOTTOM-RIGHT cell of the grid.")
    bottom_right_x, bottom_right_y = get_mouse_position_with_prompt("   ")
    cell_width = (bottom_right_x - top_left_x) / (COLS - 1)
    cell_height = (bottom_right_y - top_left_y) / (ROWS - 1)
    print(f"Cell size ≈ {cell_width:.1f} x {cell_height:.1f}")
    print("Calibration complete.\n")
    return (top_left_x, top_left_y, cell_width, cell_height)


def cell_index_to_screen(cell_index, board_geometry):
    top_left_x, top_left_y, cell_w, cell_h = board_geometry
    row = cell_index // COLS
    col = cell_index % COLS
    x = top_left_x + col * cell_w
    y = top_left_y + row * cell_h
    return int(x), int(y)


def click_anywhere_on_board(board_geometry, cell_index=0):
    x, y = cell_index_to_screen(cell_index, board_geometry)
    move_mouse(x, y)
    time.sleep(0.02)
    mouse_down(x, y)
    time.sleep(0.02)
    mouse_up(x, y)
    time.sleep(0.02)


def drag_word_path(index_path, board_geometry):
    if not index_path:
        return
    first_idx = index_path[0]
    last_idx = index_path[-1]
    start_x, start_y = cell_index_to_screen(first_idx, board_geometry)
    end_x, end_y = cell_index_to_screen(last_idx, board_geometry)
    move_mouse(start_x, start_y)
    time.sleep(0.02)
    mouse_down(start_x, start_y)
    time.sleep(0.02)
    prev_x, prev_y = start_x, start_y
    for idx in index_path[1:]:
        target_x, target_y = cell_index_to_screen(idx, board_geometry)
        drag_line(prev_x, prev_y, target_x, target_y)
        time.sleep(SLEEP_BETWEEN_LETTERS)
        prev_x, prev_y = target_x, target_y
    move_mouse(end_x, end_y)
    time.sleep(0.01)
    mouse_up(end_x, end_y)
    time.sleep(0.02)


def run_round(board_str, board_geometry):
    global stop_requested
    board1 = WordHunt(list(board_str))
    print(f"Found {len(board1.sorted)} words. Drawing them now...")
    time.sleep(0.2)
    click_anywhere_on_board(board_geometry, cell_index=0)
    stop_requested = False
    for word in board1.sorted:
        if stop_requested:
            break
        if len(word) < MIN_WORD_LEN:
            continue
        index_path = board1.answers[word]
        drag_word_path(index_path, board_geometry)
        time.sleep(SLEEP_BETWEEN_WORDS)
    print("Completed (or stopped by ESC).")

if __name__ == "__main__":
    init_stop_hotkey()
    board_geometry = load_board_geometry()
    if board_geometry is None:
        print("No calibration yet – press 'm' to calibrate the board before first use.\n")
    else:
        print("Using saved board calibration.\n")
    while True:
        command = input(
            "Enter 'm' to (re)calibrate the board,\n"
            "or enter the 4 rows of letters as 'abcd efgh ijkl mnop': \n"
            "or enter 't' for OCR"
        ).strip()
        if command == 'm':
            board_geometry = calibrate_board()
            save_board_geometry(board_geometry)
            continue
        elif command == 't':
            if board_geometry is None:
                print("No calibration yet – let's calibrate the board first.")
                board_geometry = calibrate_board()
                save_board_geometry(board_geometry)
                print("Switch to the game window – starting in 3 seconds...")
                time.sleep(3)
            board_str = ocr_board_from_screen()
            run_round(board_str, board_geometry)
            break
        if len(command) == 19:
            if board_geometry is None:
                print("No calibration yet – let's calibrate the board first.")
                board_geometry = calibrate_board()
                save_board_geometry(board_geometry)
                print("Switch to the game window – starting in 3 seconds...")
                time.sleep(3)
            board_str = "".join(command.split())
            run_round(board_str, board_geometry)
            break
        else:
            print("Invalid input format. Example: abcd efgh ijkl mnop\n")
