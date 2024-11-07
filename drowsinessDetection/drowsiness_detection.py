import cv2
import dlib  
import imutils 
from imutils import face_utils 
import scipy 
from scipy.spatial import distance 


detect = dlib.get_frontal_face_detector()  # yüzü önden tespit eden algoritmayı başlatır
predict = dlib.shape_predictor("C:\Python\mtu-2\shape_predictor_68_face_landmarks.dat")  # hazır verisetiyle yüzü işaretler


(Sağ_Başlangic, Sağ_bitis) = (36, 42)
(Sol_başlangic, Sol_bitis) = (42, 48)


# Göz üzerinde belirlenen noktalar arasında mesafe ölçümü yapan fonksiyon
def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])  #Gözün üst ve alt kenarları arasındaki mesafe (dikey mesafe).
	B = distance.euclidean(eye[2], eye[4])  #Diğer iki dikey kenarı arasındaki mesafe (yine dikey mesafe).
	C = distance.euclidean(eye[0], eye[3])  #Gözün iki yan kenarı arasındaki mesafe (yatay mesafe).

	ear = (A + B) / (2.0 * C) #formülasyon
	return ear


eşik_değer = 0.25
sayac=0
frame_sayaci = 20  # sinyal vermeden önce kaç frame sayacak 

capture=cv2.VideoCapture(0)

while (1):

	ret, frame=capture.read()
	frame = imutils.resize(frame, width=800)
	
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
	ekradaki_yuzler = detect(gray, 0) 
    

	for incelenen_yuz in ekradaki_yuzler:  
        
		shape = predict(gray, incelenen_yuz)  #yüz üzerinde noktaları işaretler
		shape = face_utils.shape_to_np(shape)  #noktaları numpy dizisine çevirme

		#gözlerin kordinatları
		Sol_göz = shape[Sol_başlangic:Sol_bitis]  
		Sağ_göz = shape[Sağ_Başlangic:Sağ_bitis]  

		solEAR = eye_aspect_ratio(Sol_göz)  
		sağEAR = eye_aspect_ratio(Sağ_göz)
		ear = (solEAR + sağEAR) / 2.0     

		sol_çerçeve = cv2.convexHull(Sol_göz)  
		sağ_çerçeve = cv2.convexHull(Sağ_göz)

		cv2.drawContours(frame, [sol_çerçeve], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [sağ_çerçeve], -1, (0, 255, 0), 1)
		
		cv2.putText(frame, f"EAR Degeri: {ear:.2f}", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
		cv2.putText(frame, f"Sayac: {sayac}", (50,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)    
		    
		if ear < eşik_değer:
			sayac = sayac + 1
			print ('Sayaç değeri: ', sayac)
			print(f"EAR: {ear:.2f}")
			
			if sayac >= frame_sayaci:
				cv2.putText(frame, "Gozunu Yoldan Ayirma", (100,550), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
		else:
			sayac = 0
		
	cv2.imshow("Kamera", frame)
	if cv2.waitKey(1) == ord("q"):
		break

cv2.destroyAllWindows()
capture.release() 