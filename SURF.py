import cv2

try:
    surf = cv2.xfeatures2d.SURF_create()
    print("SURF работает!")
except AttributeError:
    print("SURF недоступен.")
