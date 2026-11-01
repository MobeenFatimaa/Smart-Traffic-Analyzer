from pathlib import Path
from collections import defaultdict
from datetime import datetime
import csv
import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

VIDEO_PATH = BASE_DIR / "videos" / "input.mp4"
OUTPUT_PATH = BASE_DIR / "output" / "unidirectional_counted_video.mp4"
CSV_PATH = BASE_DIR / "output" / "unidirectional_vehicle_report.csv"
EVENTS_PATH = BASE_DIR / "output" / "unidirectional_vehicle_events.csv"

MODEL_NAME = "yolo11n.pt"

CONFIDENCE = 0.25
IMAGE_SIZE = 960
IOU_THRESHOLD = 0.5

# Horizontal counting line position (e.g., 75% down the frame)
LINE_Y = 0.75

# Preview window dimensions
PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 800

# Allow a vehicle to disappear briefly between observations
MAX_CROSSING_GAP = 30

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

# Speed estimation calibration factor (meters per pixel)
METERS_PER_PIXEL = 0.05  

# OpenCV BGR Colors
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
CYAN = (255, 255, 0)


# ============================================================
# 2. PREPARE OUTPUT FILES
# ============================================================

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Input video not found:\n{VIDEO_PATH}"
    )

print("=" * 60)
print("YOLO UNIDIRECTIONAL VEHICLE COUNTER & SPEED ESTIMATOR")
print("=" * 60)

print("\nLoading YOLO model...")
model = YOLO(MODEL_NAME)
print("YOLO model loaded successfully.")


# ============================================================
# 3. OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video:\n{VIDEO_PATH}"
    )

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0:
    fps = 30.0

if width <= 0 or height <= 0:
    cap.release()
    raise RuntimeError("The video has invalid dimensions.")

line_y = int(height * LINE_Y)

writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError(
        f"Could not create output video:\n{OUTPUT_PATH}"
    )

print(f"Video resolution: {width} x {height}")
print(f"Video FPS: {fps:.2f}")
print(f"Total frames: {total_frames}")
print(f"Counting line Y: {line_y}")


# ============================================================
# 4. TRACKING AND COUNTING VARIABLES
# ============================================================

last_line_side = {}
last_seen_frame = {}
counted_ids = set()

previous_positions = {}  # track_id -> (center_x, center_y, frame_number)
vehicle_speeds = {}      # track_id -> calculated speed in km/h

vehicle_counts = defaultdict(int)
total_counted = 0

vehicle_events = []
frame_number = 0


# ============================================================
# 5. DASHBOARD
# ============================================================

def draw_dashboard(frame, vehicle_counts, total_counted):
    frame_h, frame_w = frame.shape[:2]
    scale = min(frame_w / 1000, frame_h / 600)

    panel_width = min(int(350 * scale), frame_w - 20)
    panel_height = min(int(260 * scale), frame_h - 20)

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (10, 10),
        (10 + panel_width, 10 + panel_height),
        (20, 20, 20),
        -1,
    )
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    def put_text(text, x, y, font_scale=0.65, color=WHITE, thickness=2):
        cv2.putText(
            frame,
            text,
            (int(x * scale), int(y * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * scale,
            color,
            max(1, int(thickness * scale)),
            cv2.LINE_AA,
        )

    put_text("UNIDIRECTIONAL ANALYTICS", 20, 40, 0.75)
    put_text(f"Total Counted: {total_counted}", 20, 80, 0.75, GREEN)
    put_text("VEHICLE BREAKDOWN", 20, 125, 0.60, CYAN)

    y = 155
    for name in ["Car", "Motorcycle", "Bus", "Truck"]:
        count = vehicle_counts.get(name, 0)
        put_text(f"{name}: {count}", 20, y, 0.55, (235, 235, 235), 1)
        y = int(y + 25)

    return frame


# ============================================================
# 6. COUNT A VEHICLE
# ============================================================

def record_vehicle(track_id, vehicle_name, frame_number):
    global total_counted

    if track_id in counted_ids:
        return

    counted_ids.add(track_id)
    vehicle_counts[vehicle_name] += 1
    total_counted += 1

    event = {
        "Vehicle_ID": int(track_id),
        "Vehicle_Type": vehicle_name,
        "Frame_Number": frame_number,
        "Video_Time_Seconds": round(frame_number / fps, 2),
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    vehicle_events.append(event)

    print(
        f"[COUNTED] ID={track_id} | Type={vehicle_name} | Total Count={total_counted}"
    )


# ============================================================
# 7. MAIN PROCESSING LOOP
# ============================================================

print("\nStarting unidirectional video processing...")
print("Counting line direction: Crossing downwards")
print("Press Q to stop processing.\n")

try:
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_number += 1

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            classes=list(VEHICLE_CLASSES.keys()),
            verbose=False,
            iou=IOU_THRESHOLD,
        )

        # Draw the single unidirectional counting line
        cv2.line(frame, (0, line_y), (width - 1, line_y), GREEN, 3)

        active_track_ids = set()

        if (
            results
            and results[0].boxes is not None
            and results[0].boxes.id is not None
        ):
            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                active_track_ids.add(track_id)
                x1, y1, x2, y2 = map(int, box)

                vehicle_name = VEHICLE_CLASSES.get(class_id, "Vehicle")
                center_x = (x1 + x2) // 2
                bottom_y = y2

                # Speed estimation logic
                if track_id in previous_positions:
                    prev_x, prev_y, prev_frame = previous_positions[track_id]
                    frame_diff = frame_number - prev_frame
                    if frame_diff > 0:
                        pixel_distance = np.sqrt((center_x - prev_x)**2 + (bottom_y - prev_y)**2)
                        meter_distance = pixel_distance * METERS_PER_PIXEL
                        time_elapsed = frame_diff / fps
                        if time_elapsed > 0:
                            speed_mps = meter_distance / time_elapsed
                            speed_kmh = speed_mps * 3.6
                            if track_id in vehicle_speeds:
                                vehicle_speeds[track_id] = int(0.7 * vehicle_speeds[track_id] + 0.3 * speed_kmh)
                            else:
                                vehicle_speeds[track_id] = int(speed_kmh)

                previous_positions[track_id] = (center_x, bottom_y, frame_number)
                vehicle_speed = vehicle_speeds.get(track_id, 0)

                # Determine side of line (-1 above, 1 below)
                current_side = -1 if bottom_y < line_y else 1
                previous_side = last_line_side.get(track_id)
                previous_frame = last_seen_frame.get(track_id)

                if previous_side is not None:
                    frame_gap = (
                        frame_number - previous_frame
                        if previous_frame is not None
                        else MAX_CROSSING_GAP + 1
                    )
                    # Count when crossing downwards (from above -1 to below 1)
                    crossed_line = (
                        previous_side == -1
                        and current_side == 1
                        and 0 < frame_gap <= MAX_CROSSING_GAP
                    )

                    if crossed_line and track_id not in counted_ids:
                        record_vehicle(
                            track_id=track_id,
                            vehicle_name=vehicle_name,
                            frame_number=frame_number,
                        )

                last_line_side[track_id] = current_side
                last_seen_frame[track_id] = frame_number

                # Box formatting
                if track_id in counted_ids:
                    color = BLUE
                    label = f"{vehicle_name} ID:{track_id} | {vehicle_speed} km/h"
                else:
                    color = YELLOW
                    label = f"{vehicle_name} ID:{track_id} | {vehicle_speed} km/h"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text_y = max(y1 - 10, 20)
                cv2.putText(
                    frame,
                    label,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        # Cleanup old tracking memory
        for tid in list(previous_positions.keys()):
            if tid not in active_track_ids and frame_number - previous_positions[tid][2] > 60:
                del previous_positions[tid]

        # Progress info
        progress = (frame_number / total_frames) * 100 if total_frames > 0 else 0.0
        cv2.putText(
            frame,
            f"Progress: {progress:.1f}%",
            (15, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            WHITE,
            2,
            cv2.LINE_AA,
        )

        output_frame = draw_dashboard(frame.copy(), vehicle_counts, total_counted)
        writer.write(output_frame)

        # Live preview window
        preview_scale = min(PREVIEW_WIDTH / width, PREVIEW_HEIGHT / height, 1.0)
        preview_frame = cv2.resize(
            frame,
            (int(width * preview_scale), int(height * preview_scale)),
            interpolation=cv2.INTER_AREA,
        )
        preview_frame = draw_dashboard(preview_frame, vehicle_counts, total_counted)
        
        cv2.imshow("Unidirectional Vehicle Counter", preview_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nProcessing stopped by user.")
            break

finally:
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


# ============================================================
# 8. EXPORT REPORTS
# ============================================================

with open(EVENTS_PATH, "w", newline="", encoding="utf-8") as file:
    writer_csv = csv.DictWriter(
        file,
        fieldnames=["Vehicle_ID", "Vehicle_Type", "Frame_Number", "Video_Time_Seconds", "Timestamp"],
    )
    writer_csv.writeheader()
    writer_csv.writerows(vehicle_events)

with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
    writer_csv = csv.writer(file)
    writer_csv.writerow(["Vehicle_Type", "Count"])
    for v_type, count in vehicle_counts.items():
        writer_csv.writerow([v_type, count])
    writer_csv.writerow(["TOTAL", total_counted])

print("\nProcessing completed successfully.")
