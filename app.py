import cv2
import mediapipe as mp
import time
import threading
import speech_recognition as sr
from collections import Counter

print("InterviewLens - Phase 2 Started...")

# --- AI SETUP ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
mp_draw = mp.solutions.drawing_utils
recognizer = sr.Recognizer()

# --- VARIABLES ---
eye_contact = 0
total_frames = 0
spoken_text = ""
filler_words = ["um", "uh", "ah", "like", "actually", "basically"]
filler_count = 0
is_listening = True

def listen_voice():
    global spoken_text, filler_count
    with sr.Microphone() as source:
        print("Listening your voice...")
        recognizer.adjust_for_ambient_noise(source)
        while is_listening:
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                text = recognizer.recognize_google(audio).lower()
                spoken_text += " " + text
                print(f"You said: {text}")
                # Count fillers
                for word in filler_words:
                    filler_count += text.split().count(word)
            except:
                pass

# Start voice thread
voice_thread = threading.Thread(target=listen_voice, daemon=True)
voice_thread.start()

# --- CAMERA ---
cap = cv2.VideoCapture(0)
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret: break
    total_frames += 1
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

    percent = int((eye_contact/total_frames)*100) if total_frames>0 else 0
    elapsed = int(time.time() - start_time)

    # UI
    cv2.rectangle(frame, (0,0), (w,110), (0,0,0), -1)
    cv2.putText(frame, f"InterviewLens Phase 2 - LIVE {elapsed}s", (15,28), 0, 0.7, (0,255,255), 2)
    cv2.putText(frame, f"Eye Contact: {percent}% | Confidence: {85 if looking else 55}%", (15,55), 0, 0.6, (0,255,0), 2)
    cv2.putText(frame, f"Fillers (um/ah): {filler_count} | Words: {len(spoken_text.split())}", (15,80), 0, 0.6, (255,255,0), 2)
    cv2.putText(frame, f"Say something... Q to Exit & Get Report", (15,105), 0, 0.5, (255,255,255), 1)

    cv2.imshow("InterviewLens Phase 2", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        is_listening = False
        break

cap.release()
cv2.destroyAllWindows()

# --- FINAL REPORT ---
print("\n\n========== INTERVIEWLENS FINAL REPORT ==========")
print(f"Duration: {elapsed} seconds")
print(f"Eye Contact Score: {percent}%")
print(f"Total Words Spoken: {len(spoken_text.split())}")
print(f"Filler Words Used: {filler_count}")
filler_percent = (filler_count/len(spoken_text.split())*100) if spoken_text else 0
print(f"Communication Clarity: {100 - int(filler_percent*5)}%")

if percent > 70 and filler_count < 5:
    print("Overall: EXCELLENT - Ready for real interview!")
elif percent > 50:
    print("Overall: GOOD - Practice more eye contact")
else:
    print("Overall: NEEDS PRACTICE - Focus on camera")

print(f"\nWhat you said: {spoken_text[:200]}...")
print("===============================================")

# Save report
with open("Interview_Report.txt", "w") as f:
    f.write(f"InterviewLens Report\nEye Contact: {percent}%\nFillers: {filler_count}\nWords: {len(spoken_text.split())}\nSaid: {spoken_text}")

print("\nReport saved as Interview_Report.txt")