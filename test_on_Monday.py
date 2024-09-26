import torch
import cv2
import pytesseract
import numpy as np
import easyocr
import pandas as pd
from playsound import playsound

# Global variable for EasyOCR
EASY_OCR = easyocr.Reader(['en'])
OCR_TH = 0.1
ds=pd.read_csv("./data.csv")
x=pd.DataFrame(ds,index=None)
known_plates = str(x)
# Function to read known license plate numbers from data.csv


# Modify the recognize_plate_easyocr function to check if the recognized plate is known or not

# Function to generate an alarm (play a sound)
def generate_alarm():
    playsound("C:/Users/Razer/Desktop/alarm_generation/sound.mp3")  # Make sure you have an 'alarm_sound.mp3' file in the same directory

# Rest of the code remains unchanged

# -------------------------------------- function to run detection ------------------------------------------
def detectx(frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)
    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cordinates

# Function to recognize license plate numbers using Tesseract OCR
def recognize_plate_easyocr(img, coords, reader, region_threshold):
        # The same function with the corrections
        xmin, ymin, xmax, ymax = coords
        nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)]

        ocr_result = reader.readtext(nplate)
        plate_text = ocr_result[0][1]  # Extracting the license plate text from the first tuple
        print(plate_text)
        if plate_text in known_plates:
            print("Okay, right")
        else:
            generate_alarm()

        plate_num = filter_text(region=nplate, ocr_result=ocr_result, region_threshold=region_threshold)

        if len(plate_num) == 1:
            plate_num = plate_num[0].upper()
            if plate_text in known_plates:
                print("Okay, right")
            else:
                generate_alarm()
        else:
            plate_text = ""

        return plate_num

# Function to plot the BBox and results
def plot_boxes(results, frame, classes):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f"[INFO] Total {n} detections. . . ")
    print(f"[INFO] Looping through all detections. . . ")

    # Looping through the detections
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.55:  # threshold value for detection. We are discarding everything below this value
            print(f"[INFO] Extracting BBox coordinates. . . ")

            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            text = classes[int(labels[i])]

            coords = [x1, y1, x2, y2]

            plate_num = recognize_plate_easyocr(img=frame, coords=coords, reader=EASY_OCR, region_threshold=OCR_TH)
            

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # BBox
            cv2.rectangle(frame, (x1, y1 - 20), (x2, y1), (0, 255, 0), -1)  # for text label background
            cv2.putText(frame, f'{plate_num}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame

# Function to filter out wrong detections
def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0] * region.shape[1]
    plate = []
    print(ocr_result)
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        if length * height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate

# Main function
def main(img_path=None, vid_path=None, vid_out=None):
    print(f"[INFO] Loading model... ")
    model = torch.hub.load('C:/Users/Razer/Downloads/yolov5_deploy-main/yolov5_deploy-main/yolov5_deploy/yolov5_data/content/yolov5', 'custom', source='local', path=r'C:\Users\Razer\Downloads\yolov5_deploy-main\yolov5_deploy-main\yolov5_deploy\yolov5_data\content\yolov5\yolov5s.pt', force_reload=True)
    classes = model.names
    
    if img_path:
        print(f"[INFO] Working with image: {img_path}")
        img_out_name = f"./output/result_{img_path.split('/')[-1]}"
        
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = detectx(frame, model=model)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame = plot_boxes(results, frame, classes=classes)
        
        
        cv2.namedWindow("img_only", cv2.WINDOW_NORMAL)

        while True:
            cv2.imshow("img_only", frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                print(f"[INFO] Exiting. . . ")
                cv2.imwrite(f"{img_out_name}", frame)
                break

    elif vid_path:
        print(f"[INFO] Working with video: {vid_path}")
        cap = cv2.VideoCapture(vid_path)

        if vid_out:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(vid_out, codec, fps, (width, height))

        frame_no = 1
        cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            if ret and frame_no % 1 == 0:
                print(f"[INFO] Working with frame {frame_no} ")

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detectx(frame, model=model)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                frame = plot_boxes(results, frame, classes=classes)

                cv2.imshow("vid_out", frame)
                if vid_out:
                    print(f"[INFO] Saving output video. . . ")
                    out.write(frame)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                frame_no += 1

        print(f"[INFO] Cleaning up. . . ")
        out.release()
        cv2.destroyAllWindows()

# Call the main function and pass the image path
if __name__ == "__main__":
   main(img_path=r"C:\Users\Razer\Downloads\yolov5_deploy-main\yolov5_deploy-main\Car2.jpg")  # Replace with your image path
