import cv2
import time
import sys

# COR DAS CLASSES
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0), (0, 0, 255)]
telacheia = True

#CARREGAR AS CLASSES
class_name = []
with open(r"coco.names", "r") as f:
  class_name = [cname.split()[0] for cname in f.readlines()]

#CAPTURA DO VIDEO
winName = 'Captura'
if telacheia:
  cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
  cv2.setWindowProperty(winName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

#webcam = cv2.VideoCapture(r"Teste1.mp4")
webcam = cv2.VideoCapture(r"Teste2.mov")
#webcam = cv2.VideoCapture(0)

#CARREGANDO PESOS DA REDE NEURAL
net = cv2.dnn.readNet(r"yolov4-tiny.cfg", r"yolov4-tiny.weights")


#SETANDO REDE NEURAL
model = cv2.dnn.DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)


if webcam.isOpened():
  validacao, frame = webcam.read()
else:
  print("Deu ruim")
  sys.exit()

while validacao:
  validacao, frame = webcam.read()

  #COMEÇAR DETECÇÃO
  inicio = time.time()

  #DETECÇÃO
  classes, scores, boxes = model.detect(frame, 0.1, 0.2)

  #FIM DETECÇÃO
  fim = time.time()
  #scores = [round(score, 2) for score in scores]

  for (classid, score, box) in zip(classes, scores, boxes):
    #GERANDO UMA COR
    color = COLORS[int(classid) % len(COLORS)]
    #PEGANDO NOME E SCORE DA ACURACIA PELO ID
    label = f"{class_name[classid]} : {round(score,1)}"

    #DESENHADO O BOX DE DETECÇÃO
    cv2.rectangle(frame, box, color, 2)

    #ESCREVENDO NOME EM CIMA DO OBJETO
    cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  #CALCULANDO FPS
  fps_label = f"FPS : {round((1.0/(fim - inicio)), 2)}"

  #ESCREVENDO FPS
  cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
  cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

  #MOSTRANDO A IMAGEM
  cv2.imshow(winName, frame)
  key = cv2.waitKey(8)

  if key == 27: # ESC
    break

webcam.release()
cv2.destroyAllWindows()
