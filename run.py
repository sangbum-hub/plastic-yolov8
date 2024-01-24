import os
import cv2
from ultralytics import YOLO
from datetime import datetime
import argparse


# YOLO 모델 초기화 함수
def initialize_yolo_model(model_path='./runs/detect/train2/weights/best.pt'):
    model = YOLO(model_path)
    return model


# 비디오 또는 웹캠 캡처 합수
# use_video_file 값이 True : 동영상 파일 실행 | False : 웹캠 실행  
def open_video_capture(use_video_file, video_path):
    if use_video_file:
        return cv2.VideoCapture(video_path)
    else:
        return cv2.VideoCapture(0)


# 감지 정보를 로그에 기록하는 함수
def log_detection(log_file, box, label_cls, model_names):
    center_x = int((box[0] + box[2]) / 2)
    center_y = int((box[1] + box[3]) / 2)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"{current_time} - bounding_box: {box}, label: {model_names[int(label_cls)]}, center_point: ({center_x}, {center_y})\n")


# 프레임에 라벨 및 좌표값을 주석으로 표시하는 함수
def display_annotated_frame(frame, result, model):
    annotated_frame = result[0].plot()
    detections = result[0].boxes

    if len(detections) > 0:
        for box, label_cls in zip(detections.xyxy, detections.cls):
            print("bounding_box:", box)
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)
            print("label:", model.names[int(label_cls)])

            cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)
            center_text = f"({center_x}, {center_y})"
            name_text = f"{model.names[int(label_cls)]}"
            cv2.putText(annotated_frame, center_text, (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(annotated_frame, name_text, (center_x + 10, center_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 추론", annotated_frame)
    return cv2.waitKey(1) & 0xFF == ord("q")


# 비디오 리소스 해제 함수
def release_resources(vc):
    vc.release()
    cv2.destroyAllWindows()


# main 함수
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-video", dest="use_video_file",action="store", default=False)
    parser.add_argument("-path", dest="video_path",action="store", default="./video/01.mp4")
    parser.add_argument("-log", dest="log_save",action="store", default="y")

    args = parser.parse_args()

    model = initialize_yolo_model()
    vc = open_video_capture(use_video_file=args.use_video_file, video_path=args.video_path)

    log_detections = args.log_save.lower() == 'y'

    if log_detections:

        log_file_path = "detection_logs.txt"

        if not os.path.exists(log_file_path):
            # log파일 없을 경우 생성 
            with open(log_file_path, "w"):
                pass
        
        log_file = open(log_file_path, "a")

    while vc.isOpened():
        success, frame = vc.read()

        if success:
            result = model(frame, conf=0.3)

            if log_detections:
                for box, label_cls in zip(result[0].boxes.xyxy, result[0].boxes.cls):
                    log_detection(log_file, box, label_cls, model.names)

            if display_annotated_frame(frame, result, model):
                break
        else:
            break

    if log_detections:
        log_file.close()
        print(f"감지 정보가 {log_file_path}에 저장되었습니다.")

    release_resources(vc)


if __name__=="__main__":
    main()



