import cv2
import numpy as np
import torch
from ultralytics import YOLO
import torchreid
from sklearn.metrics.pairwise import cosine_similarity
import pytesseract
from collections import deque

# Configuration
SIMILARITY_THRESHOLD = 0.7
MAX_MISSED_FRAMES = 20
REID_HISTORY_LENGTH = 5
LOST_PLAYER_BUFFER = 30
MIN_DETECTION_CONFIDENCE = 0.5
PLAYER_CLASS_ID = 2
MIN_NUMBER_CONFIDENCE = 0.8

# Setup Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load YOLO model
yolo = YOLO("static/assets/best.pt")

# Load ReID model
reid_model = torchreid.models.build_model('osnet_x1_0', num_classes=1, loss='softmax')
torchreid.utils.load_pretrained_weights(reid_model, "static/reid_model/osnet_x1_0_imagenet.pth")
reid_model.eval()
if torch.cuda.is_available():
    reid_model = reid_model.cuda()


def extract_reid_feature(image):
    image = cv2.resize(image, (128, 256))
    image = image[:, :, ::-1].astype(np.float32) / 255.0
    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
    if torch.cuda.is_available():
        image = image.cuda()
    with torch.no_grad():
        features = reid_model(image)
    return features.cpu().numpy().flatten()


def extract_jersey_number(image):
    h, w = image.shape[:2]
    region = image[max(0, h // 4):min(h, h // 2 + h // 4), :]
    best_number = "?"
    best_confidence = 0.0

    for thresh_val in [120, 150, 180]:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY_INV)

        config = "--psm 8 -c tessedit_char_whitelist=0123456789"
        data = pytesseract.image_to_data(thresh, config=config, output_type=pytesseract.Output.DICT)

        for i in range(len(data['text'])):
            text = data['text'][i]
            conf = float(data['conf'][i]) / 100 if data['conf'][i] else 0
            digits = ''.join(filter(str.isdigit, text))
            if digits and conf > best_confidence:
                best_number = digits
                best_confidence = conf

    return best_number, best_confidence


def classify_team(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
    red1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 70, 50), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)

    blue_pixels = cv2.countNonZero(blue)
    red_pixels = cv2.countNonZero(red)

    if blue_pixels > red_pixels and blue_pixels > 50:
        return "Blue"
    elif red_pixels > blue_pixels and red_pixels > 50:
        return "Red"
    return "Unknown"


class PlayerTrack:
    def __init__(self, pid, feature, team, number, position, frame_count):
        self.id = pid
        self.features = deque([feature], maxlen=REID_HISTORY_LENGTH)
        self.team = team
        self.number = number
        self.number_confidence = 0.0
        self.last_position = position
        self.positions = deque([position], maxlen=5)
        self.missed_frames = 0
        self.active = True
        self.first_detected = frame_count
        self.last_updated = frame_count

    def update(self, feature, position, team, number, number_confidence, frame_count):
        self.features.append(feature)
        self.last_position = position
        self.positions.append(position)
        self.team = team if team != "Unknown" else self.team
        if number_confidence > self.number_confidence:
            self.number = number
            self.number_confidence = number_confidence
        self.missed_frames = 0
        self.last_updated = frame_count

    @property
    def feature(self):
        return np.mean(self.features, axis=0)

    def predict_position(self):
        if len(self.positions) < 2:
            return self.last_position
        dx = self.positions[-1][0] - self.positions[-2][0]
        dy = self.positions[-1][1] - self.positions[-2][1]
        return (self.last_position[0] + dx, self.last_position[1] + dy)


class FootballTracker:
    def __init__(self):
        self.player_tracks = []
        self.lost_players = []
        self.next_id = 1
        self.frame_count = 0
        self.used_ids = set()

    def process_frame(self, frame):
        self.frame_count += 1
        results = yolo(frame)[0]

        player_detections = []
        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box[:6]
            if conf < MIN_DETECTION_CONFIDENCE or int(cls) != PLAYER_CLASS_ID:
                continue
            player_detections.append((int(x1), int(y1), int(x2), int(y2), float(conf)))

        for x1, y1, x2, y2, conf in player_detections:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            feature = extract_reid_feature(crop)
            number, number_conf = extract_jersey_number(crop)
            team = classify_team(crop)
            position = ((x1 + x2) // 2, (y1 + y2) // 2)

            best_match = self.find_best_match(feature, position, number, number_conf, team)

            if best_match and best_match['score'] >= SIMILARITY_THRESHOLD:
                player = best_match['player']
                player.update(feature, position, team, number, number_conf, self.frame_count)
                player_id = player.id
                if player in self.lost_players:
                    self.lost_players.remove(player)
            else:
                player_id = self.get_unique_id()
                new_player = PlayerTrack(player_id, feature, team, number, position, self.frame_count)
                new_player.number_confidence = number_conf
                self.player_tracks.append(new_player)

            self.draw_player_info(frame, x1, y1, x2, y2, player_id, number, team)

        self.update_tracks()
        return frame

    def find_best_match(self, feature, position, number, number_conf, team):
        best_match = {'player': None, 'score': -1}
        for player in self.player_tracks:
            if not player.active:
                continue
            dist = np.linalg.norm(np.array(position) - np.array(player.predict_position()))
            if dist > 200:
                continue
            appearance_sim = cosine_similarity([feature], [player.feature])[0][0]
            pos_sim = max(0, 1 - (dist / 200))
            number_sim = 1.0 if number == player.number else -0.5
            team_sim = 1.0 if team == player.team else 0.0
            score = (0.5 * appearance_sim + 0.3 * pos_sim + 0.1 * number_sim + 0.1 * team_sim)
            if score > best_match['score']:
                best_match = {'player': player, 'score': score}
        return best_match

    def get_unique_id(self):
        while True:
            new_id = self.next_id
            self.next_id += 1
            if new_id not in self.used_ids:
                self.used_ids.add(new_id)
                return new_id

    def update_tracks(self):
        for player in self.player_tracks:
            if player.last_updated < self.frame_count:
                player.missed_frames += 1
                if player.missed_frames > MAX_MISSED_FRAMES:
                    player.active = False
                    if player.missed_frames <= LOST_PLAYER_BUFFER:
                        self.lost_players.append(player)
        self.player_tracks = [p for p in self.player_tracks if p.active]
        self.lost_players = [p for p in self.lost_players if (self.frame_count - p.last_updated) <= LOST_PLAYER_BUFFER]

    def draw_player_info(self, frame, x1, y1, x2, y2, player_id, number, team):
        color = (0, 255, 0) if team == "Blue" else (0, 0, 255) if team == "Red" else (200, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"ID:{player_id}"
        if number != "?":
            text += f" #{number}"
        text += f" {team}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    tracker = FootballTracker()

    cv2.startWindowThread()
    cv2.namedWindow("Football Tracking", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = tracker.process_frame(frame)
        out.write(processed_frame)

        cv2.imshow("Football Tracking", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processing complete. Output saved to {output_path}")


if __name__ == "__main__":
    process_video("static/assets/15sec_input_720p.mp4", "output_with_consistent_ids.mp4")
