import cv2
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self, minDetectionCon=0.5):

        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection()

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        self.results = self.faceDetection.process(imgRGB)

        bboxes = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # Get the bounding box come from the class
                bboxC = detection.location_data.relative_bounding_box
                # Get the image width and height from the video
                ih, iw, ic = img.shape
                # Calculate the coordinates from the detections
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxes.append([id, bbox, detection.score])
                img = self.fancyDraw(img, bbox)

                cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_PLAIN, 2,(255, 0, 0), 1)

        return img, bboxes

    def fancyDraw(self, img, bbox, color=(255, 0, 0), l=10, t=5):
        # Coordinates of the 4 points of the rectangle
        x, y, w, h = bbox
        x1, y1 = x+w, y+h

        # Rectangle
        cv2.rectangle(img, bbox, (255, 0, 0), 1)

        # Top Left
        cv2.line(img, (x, y), (x+l, y), color, t)
        cv2.line(img, (x, y), (x, y+l), color, t)
        # Top Right
        cv2.line(img, (x1, y), (x1 - l, y), color, t)
        cv2.line(img, (x1, y), (x1, y + l), color, t)
        # Bottom Left
        cv2.line(img, (x, y1), (x + l, y1), color, t)
        cv2.line(img, (x, y1), (x, y1 - l), color, t)
        # Bottom Right
        cv2.line(img, (x1, y1), (x1 - l, y1), color, t)
        cv2.line(img, (x1, y1), (x1, y1 - l), color, t)

        return img
    
def main():
    # Webcam
    cap = cv2.VideoCapture(0)
    pTime = 0

    # Initialize the FaceDetector Class
    detector = FaceDetector()

    # Run the video
    while cap.isOpened():
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)

        # Get the fps
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        # Display the video
        cv2.imshow("Image", img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()