from cv2 import cv2
import dlib

from scipy.spatial import distance
from pygame import mixer
import time
import speech_recognition as sr
from concurrent.futures import ThreadPoolExecutor

r = sr.Recognizer()

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
mixer.init()
sound = mixer.Sound('alarm.mp3')
score = 0
f = 0
g = 0


def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A + B) / (2.0 * C)
    return ear_aspect_ratio


cap = cv2.VideoCapture(0)
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def wait_function():
    with sr.Microphone() as source:
        print("Speak Anything :")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            # print("You said : {}".format(text))
        except:
            print("Sorry could not recognize what you said")
        return (text)


def camera(score):
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)

    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []

        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = n + 1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = n + 1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)

        EAR = (left_ear + right_ear) / 2
        EAR = round(EAR, 2)
        # print(EAR)
        if EAR < 0.24:
            """cv2.putText(frame,"DROWSY",(20,100),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),4)
			cv2.putText(frame,"Are you Sleepy?",(20,400),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
			print("Drowsy")"""
            score = score + 0.4

        else:
            score = score - 0.4
        if (score < 0):
            score = 0
        cv2.putText(frame, 'score:' + str(round(score)), (20, 400), font, 1, (255, 0, 0), 1, cv2.LINE_AA)

    return frame, score


while True:
    frame, score = camera(score)

    if (score > 15):
        # if the person is feeling sleepy so we beep the alarm
        try:
            if (f == 0):
                f = 1
                sound.play(-1)
            while True:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(wait_function)

                    while True:
                        if (future.running()):
                            frame, score = camera(score)
                            cv2.putText(frame, 'ALARM', (150, 75), font, 4, (0, 0, 255), 4, cv2.LINE_AA)
                            cv2.imshow("Drowsinness dectector", frame)
                            if cv2.waitKey(20) & 0xFF == ord('q'):
                                break
                        elif (future.done()):
                            print(future.result())
                            break
                if (future.result() == 'I am okay' and score < 10 and f == 1):
                    sound.stop()
                    break


        except:  # isplaying = False
            pass
    

    cv2.imshow("Drowsinness dectector", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
