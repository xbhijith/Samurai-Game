import cv2
import mediapipe as mp
import pygame
import numpy as np
import threading
import time

class VideoCaptureThread:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        self.ret = False
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            if self.cap.isOpened():
                self.ret, self.frame = self.cap.read()
            time.sleep(0.01)

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

def calculate_angle(a, b, c):
    ab = np.array(b) - np.array(a)
    bc = np.array(c) - np.array(b)
    angle = np.arctan2(bc[1], bc[0]) - np.arctan2(ab[1], ab[0])
    return np.abs(np.degrees(angle))

def display_Player1_score(screen, font, score):
    score_surface = font.render(f'Score: {score}', True, (255, 255, 255))
    screen.blit(score_surface, (10, 10))

def display_Player2_score(frame, font, Player2_score):
    score_surface = font.render(f'Score: {Player2_score}', True, (255, 255, 255))
    frame.blit(score_surface, (10, 10))

Player1_ip = input("Enter the IP address: ")
Player1_port = input("Enter the port: ")
cap_thread = VideoCaptureThread(f'http://{Player1_ip}:{Player1_port}/video')
Player2_cap_thread = VideoCaptureThread(0)

pygame.init()

camera_screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Player1 Frame")

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

running = True
score = 0
Player2_score = 0
angle_reached = False
previous_wrist_position = None
font = pygame.font.Font(None, 36)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    ret, frame = cap_thread.read()
    Player2_ret, Player2_frame = Player2_cap_thread.read()

    if ret:
        frame = cv2.resize(frame, (400, 300))
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        skeleton_frame = np.zeros_like(frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(skeleton_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
            wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

            angle = calculate_angle(
                (shoulder.x, shoulder.y),
                (elbow.x, elbow.y),
                (wrist.x, wrist.y)
            )

            if previous_wrist_position is not None:
                wrist_x, wrist_y = wrist.x, wrist.y
                prev_x, prev_y = previous_wrist_position

            if angle < 30 and not angle_reached:
                score += 1
                angle_reached = True
            elif angle >= 30:
                angle_reached = False

            previous_wrist_position = (wrist.x, wrist.y)

        camera_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        camera_frame = np.rot90(camera_frame, k=1)
        camera_frame_surface = pygame.surfarray.make_surface(camera_frame)
        camera_screen.blit(camera_frame_surface, (0, 0))
        display_Player1_score(camera_screen, font, score)
        pygame.display.flip()

        cv2.imshow('Player1 Skeleton View', skeleton_frame)

        if Player2_ret:
            Player2_frame = cv2.resize(Player2_frame, (400, 300))
            Player2_image_rgb = cv2.cvtColor(Player2_frame, cv2.COLOR_BGR2RGB)
            Player2_results = pose.process(Player2_image_rgb)

            if Player2_results.pose_landmarks:
                Player2_shoulder = Player2_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                Player2_elbow = Player2_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
                Player2_wrist = Player2_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

                Player2_angle = calculate_angle(
                    (Player2_shoulder.x, Player2_shoulder.y),
                    (Player2_elbow.x, Player2_elbow.y),
                    (Player2_wrist.x, Player2_wrist.y)
                )

                if Player2_angle < 30 and not angle_reached:
                    Player2_score += 1
                    angle_reached = True
                elif Player2_angle >= 30:
                    angle_reached = False

                Player2_skeleton_frame = np.zeros_like(Player2_frame)
                mp_drawing.draw_landmarks(Player2_skeleton_frame, Player2_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                Player2_skeleton_frame = cv2.cvtColor(Player2_skeleton_frame, cv2.COLOR_BGR2RGB)
                cv2.imshow('Player2 Skeleton View', Player2_skeleton_frame)

            Player2_frame = cv2.putText(Player2_frame, f'Score: {Player2_score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Player2 Frame', Player2_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        running = False

    time.sleep(0.01)

cap_thread.release()
Player2_cap_thread.release()
pose.close()
cv2.destroyAllWindows()
pygame.quit()
