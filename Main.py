import cv2
from cvzone.FaceDetectionModule import FaceDetector

# Carrega o classificador de gatos
catFaceCascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')

# Inicializa o detector de rostos humanos
detector = FaceDetector()

# Inicia a webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converte o frame para escala de cinza para detecção de gatos
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cats = catFaceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Desenha retângulo para gatos
    for (x, y, w, h) in cats:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 128, 0), 2)

    # Detecta rostos humanos
    frame, bboxes = detector.findFaces(frame, draw=True)

    cv2.imshow('Detecção de Gatos e Rostos Humanos', frame)

    if cv2.waitKey(1) == 27:  # Pressione 'Esc' para sair
        break

cap.release()
cv2.destroyAllWindows()