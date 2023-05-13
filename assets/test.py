import smoothingmodule
import cv2

im = smoothingmodule.RetouchingImg("Orig1.png","shape_predictor_68_face_landmarks.dat")

cv2.imwrite("result.png", im)