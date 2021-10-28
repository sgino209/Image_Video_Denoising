import cv2
import numpy as np

interactive_mode = False

print('Started')

cap = cv2.VideoCapture('demo1.mp4')
cap_fps = int(cap.get(cv2.CAP_PROP_FPS))
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'avc1')
outname = 'demo1_res.mp4'

out = cv2.VideoWriter(outname, fourcc, cap_fps, (6*cap_width, cap_height))

while cap.isOpened():
    
    ret, frame = cap.read()
    
    if ret==True:
       
        ret, thresh = cv2.threshold(frame, 250, 255, cv2.THRESH_BINARY)  # pixels > THR ==> 255, pixels < 255 ==> 0 
        
        median = cv2.medianBlur(frame, 35)
        gaussian = cv2.GaussianBlur(frame, (35,35), 0)
        bilateral = cv2.bilateralFilter(frame, 35, 0, 0)
        thresh_median7 = cv2.medianBlur(thresh,7)
        thresh_median15 = cv2.medianBlur(thresh, 15)

        cv2.putText(frame, "Original", (int(cap_width/2)-50,60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv2.putText(median, "Median 35", (int(cap_width/2)-50,60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv2.putText(gaussian, "Gaussian 35", (int(cap_width/2)-50,60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv2.putText(bilateral, "Bilateral 35", (int(cap_width/2)-50,60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv2.putText(thresh_median7, "Thresh + Median 7", (int(cap_width/2)-50,60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv2.putText(thresh_median15, "Thresh + Median 15", (int(cap_width/2)-50,60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        
        compare = np.concatenate((frame, median, gaussian, bilateral, thresh_median7, thresh_median15), axis=1)

        out.write(compare)

        if interactive_mode:
            cv2.imshow('compare', compare)        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        break

cap.release()
out.release()

if interactive_mode:
    cv2.destroyAllWindows()

print('Completed Successfully')

