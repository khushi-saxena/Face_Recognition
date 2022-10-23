import cv2
import numpy as np
import os


####KNN ALGORITHM CODE#####

def dist(X1,X2):
    return np.sqrt(np.sum((X1-X2)**2))
def knn(X,Query, k = 5):
    m = X.shape[0]
#     print(Query.shape)
    vals = []
    for i in range(m):
        xi=X[i,:-1]
        yi=X[i,-1]
#         print(Query[i].shape,X[i].shape)
        d = dist(Query, xi)
        vals.append((d,yi))
    
    vals = sorted(vals,key= lambda x:x[0])[:k]
    vals = np.asarray(vals)
    
    new_vals = np.unique(vals[:,1],return_counts = True)
#     print(new_vals)
    index = new_vals[1].argmax()
    pred = new_vals[0][index]
    return pred
##############################


cap=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip=0
face_data=[]
dataset_path='./data/'
labels=[]

class_id=0
names={}
#face_section=np.zeros((100,100),dtype='uint8') 
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id]=fx[:-4]
        print("loaded "+fx)
        data_item=np.load(dataset_path+fx)
        face_data.append(data_item)

        target=class_id*np.ones((data_item.shape[0],))
        class_id+=1
        labels.append(target)

face_dataset=np.concatenate(face_data,axis= 0)
face_labels=np.concatenate(labels,axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

trainset=np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)


while True:
    ret,frame=cap.read()
    
    if ret==False:
        continue
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    faces=sorted(faces,key=lambda f:f[2]*f[3])


    for face in faces:
        x,y,w,h=face

        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))

        out=knn(trainset,face_section.flatten())

        pred_name=names[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    cv2.imshow("Faces",frame)
    
    key_pressed=cv2.waitKey(1)& 0xFF
    if key_pressed== ord('s'):
        break

cap.release()
cv2.destroyAllWindows()    

