import cv2
import numpy as np

mode = 'production'  # 'interactive' / 'exploration' / 'production'
keep_only_left_side = False

print('Started')

# Prolog:
cap = cv2.VideoCapture('demo1.mp4')
cap_fps = int(cap.get(cv2.CAP_PROP_FPS))
cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'avc1')
outname = 'demo1_res.mp4'

images_num = 6
if mode == 'production':
    images_num = 3
        
if keep_only_left_side:
    cap_width = int(cap_width/2) 

out = cv2.VideoWriter(outname, fourcc, cap_fps, (images_num*cap_width, cap_height), 0)

# Main Course:
while cap.isOpened():
    
    ret, frame_rgb = cap.read()
 
    if ret==True:
   
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)

        if keep_only_left_side:
            frame = frame[:, :cap_width]

        # B/W Thresholding (pixels > THR ==> 255, pixels < 255 ==> 0):
        _, thresh = cv2.threshold(frame, int(0.9*frame.max()), 255, cv2.THRESH_BINARY)
       
        # Filtering: 
        if mode != 'production':
            median = cv2.medianBlur(frame, 35)
            gaussian = cv2.GaussianBlur(frame, (35,35), 0)
            bilateral = cv2.bilateralFilter(frame, 35, 0, 0)
            thresh_median7 = cv2.medianBlur(thresh,7)
        thresh_median15 = cv2.medianBlur(thresh, 15)

        # Visualization:
        cv2.putText(frame, "Original", (int(cap_width/2)-50,60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv2.putText(thresh, "Threshold", (int(cap_width/2)-50,60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv2.putText(thresh_median15, "Thresh + Median 15", (int(cap_width/2-50),60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        if mode != 'production':
            cv2.putText(median, "Median 35", (int(cap_width/2)-50,60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
            cv2.putText(gaussian, "Gaussian 35", (int(cap_width/2)-50,60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
            cv2.putText(bilateral, "Bilateral 35", (int(cap_width/2)-50,60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
            cv2.putText(thresh_median7, "Thresh + Median 7", (int(cap_width/2)-100,60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        
        _, thresh_median15_bw = cv2.threshold(thresh_median15, int(0.6*thresh_median15.max()), 255, cv2.THRESH_BINARY)
 
        if mode == 'production':
            compare = np.concatenate((frame, thresh, thresh_median15_bw), axis=1)
        else:
            compare = np.concatenate((frame, median, gaussian, bilateral, thresh_median7, thresh_median15_bw), axis=1)

        out.write(compare)

        # Interactive:
        if mode == 'interactive':
            cv2.imshow('compare', compare)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        break

# Epilog:
cap.release()
out.release()

if mode == 'interactive':
    cv2.destroyAllWindows()

print('Completed Successfully')

