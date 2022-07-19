import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

while True:
    # Captura frame-por-frame
    ok, frame = video_capture.read()

    imagem_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    deteccoes = face_detector.detectMultiScale(imagem_cinza, minSize=(100,100))

    # Desenha o retângulo sobre as faces
    for (x, y, w, h) in deteccoes:
        print(w, h)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    # Mostra o resultado no vídeo
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()