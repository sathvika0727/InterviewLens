import cv2
import mediapipe as mp
import time

print("InterviewLens Started...")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
start_time = time.time()
eye_contact = 0
total = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    total += 1
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    looking = False
    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            nose = face.landmark[1]
            if 0.4 < nose.x < 0.6 and 0.35 < nose.y < 0.65:
                looking = True
                eye_contact += 1
            mp_draw.draw_landmarks(frame, face, mp_face_mesh.FACEMESH_CONTOURS)

    percent = int((eye_contact/total)*100) if total>0 else 0
    conf = 85 if looking else 60

    cv2.rectangle(frame, (0,0), (w,95), (0,0,0), -1)
    cv2.putText(frame, "InterviewLens - LIVE", (20,30), 0, 0.8, (0,255,255), 2)
    cv2.putText(frame, f"Eye Contact: {percent}%", (20,60), 0, 0.7, (0,255,0), 2)
    cv2.putText(frame, f"Confidence: {conf}%", (20,85), 0, 0.7, (255,255,0), 2)
    cv2.putText(frame, "Press Q to Exit", (20, h-15), 0, 0.5, (255,255,255), 1)

    cv2.imshow("AI Interview - InterviewLens", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n--- FINAL REPORT ---")
print(f"Eye Contact: {percent}%")
if percent > 75:
    print("Feedback: Excellent! Baga chusav")
elif percent > 50:
    print("Feedback: Good, koncham improve")
else:
    print("Feedback: Camera vaipu chudali")
print(f"Time: {int(time.time()-start_time)} sec")