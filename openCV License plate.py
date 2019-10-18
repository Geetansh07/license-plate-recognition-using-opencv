#!/usr/bin/env python
# coding: utf-8

# In[17]:


import cv2

#making a cascade
license_cascade = cv2.CascadeClassifier("C:\\Users\\geetk\\Downloads\\opencv2\\Library\\etc\\haarcascades\\haarcascade_russian_plate_number.xml")


#image of a car's number plate
img = cv2.imread("G:\\Let's do this\\Open CV\\liscense plate recognition\\10.png")

#convert this image to Gray
gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


#search for the number 
num = license_cascade.detectMultiScale(gray_image,scaleFactor = 1.05,minNeighbors = 5)


#make a rectangle around that image
for x,y,w,h in num:
    img = cv2.rectangle(img, (x,y) ,(x+w, y+2*h),(0,255,0),10)
    
resize = cv2.resize(img,(400,400))    
cv2.imshow("Gray",resize)

cv2.waitKey(0)

cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




