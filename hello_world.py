from pathlib import Path

import cv2  # opencv - display the video stream
import depthai  # access the camera and its data packets
import numpy as np

device = depthai.Device('', False)

# Create the pipeline using the 'previewout' stream, establishing the first connection to the device.
pipeline = device.create_pipeline(config={
    'streams': ['previewout', 'metaout'],
    'ai': {
        "blob_file": str(Path('tiny-yolo-v3.blob').resolve().absolute()),
        "blob_file_config": str(Path('tiny-yolo-v3.json').resolve().absolute()),
    }
})

if pipeline is None:
    raise RuntimeError('Pipeline creation failed!')

detections = []

red_lower = np.array([0, 150, 150], np.uint8) 
red_upper = np.array([10, 255, 255], np.uint8) 

green_lower = np.array([74, 143, 171], np.uint8) 
green_upper = np.array([94, 163, 251], np.uint8) 

while True:
    # Retrieve data packets from the device.
    # A data packet contains the video frame data.
    nnet_packets, data_packets = pipeline.get_available_nnet_and_data_packets()

    for nnet_packet in nnet_packets:
        detections = list(nnet_packet.getDetectedObjects())
        # print(detections)

    for packet in data_packets:
        # By default, DepthAI adds other streams (notably 'meta_2dh'). Only process `previewout`.
        if packet.stream_name == 'previewout':
            data = packet.getData()
            # change shape (3, 300, 300) -> (300, 300, 3)
            data0 = data[0, :, :]
            data1 = data[1, :, :]
            data2 = data[2, :, :]
            frame = cv2.merge([data0, data1, data2])
            hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            kernal = np.ones((5, 5), "uint8") 
            imCrop = frame
            img_h = frame.shape[0]
            img_w = frame.shape[1]

            for detection in detections:
                print(detection.label)
                if(detection.label == 9):
                    pt1 = int(detection.x_min * img_w), int(detection.y_min * img_h)
                    pt2 = int(detection.x_max * img_w), int(detection.y_max * img_h)

                    x, y = pt1
                    x_w, y_h = pt2
                    imCrop = hsvFrame[y:y_h, x:x_w]
                
                    red_mask = cv2.inRange(imCrop, red_lower, red_upper)
                    green_mask = cv2.inRange(imCrop, green_lower, green_upper) 
                    #  # For red color 
                    red_mask = cv2.dilate(red_mask, kernal) 
                    res_red = cv2.bitwise_and(imCrop, imCrop,  
                                            mask = red_mask) 
                    
                    # For green color 
                    green_mask = cv2.dilate(green_mask, kernal) 
                    res_green = cv2.bitwise_and(imCrop, imCrop, 
                                                mask = green_mask) 
                    
                    # Creating contour to track red color 
                    contours, hierarchy = cv2.findContours(red_mask, 
                                                        cv2.RETR_TREE, 
                                                        cv2.CHAIN_APPROX_SIMPLE) 
                    
                    for pic, contour in enumerate(contours): 
                        area = cv2.contourArea(contour) 
                        if(area > 30): 
                            x_, y_, w, h = cv2.boundingRect(contour) 
                            # img = cv2.rectangle(imCrop, (x_, y_),  
                            #                         (x_ + w, y_ + h),  
                            #                         (0, 0, 255), 2) 
                            
                            cv2.putText(frame, "Red", (x, y-20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                        (0, 0, 255), 2)     
                            print("Traffic Light: RED")

                    # Creating contour to track green color 
                    contours, hierarchy = cv2.findContours(green_mask, 
                                                        cv2.RETR_TREE, 
                                                        cv2.CHAIN_APPROX_SIMPLE) 
                    
                    for pic, contour in enumerate(contours): 
                        area = cv2.contourArea(contour) 
                        if(area > 3): 
                            x_, y_, w, h = cv2.boundingRect(contour) 
                            # img = cv2.rectangle(imCrop, (x_, y_),  
                            #                         (x_ + w, y_ + h), 
                            #                         (0, 255, 0), 2) 
                            
                            cv2.putText(frame, "Green", (x, y-20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                        (0, 255, 0), 2)  
                            print("Traffic Light: GREEN")

                    cv2.rectangle(frame, pt1, pt2, (0, 0, 255), 2)
                

            cv2.imshow('previewout', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# The pipeline object should be deleted after exiting the loop. Otherwise device will continue working.
# This is required if you are going to add code after exiting the loop.
del device
