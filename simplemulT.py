import cv2
from random import randint

trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

def createTracker():
    tracker = cv2.TrackerKCF_create() # creating instance of tracker
    return tracker

video_path = "run.mp4"

cap = cv2.VideoCapture(video_path) # creating video reader

_, first_f = cap.read()

bboxes = [] 
colors = []

while True: # looping to mark initial detections
    bbox = cv2.selectROI('MultiTracker', first_f)
    bboxes.append(bbox)
    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    print("Press q to quit selecting boxes and start tracking")
    print("Press any other key to select next object")
    k = cv2.waitKey(0) & 0xFF
    if (k == 113):  # q is pressed
        break

print('Selected bounding boxes {}'.format(bboxes))

# Create MultiTracker object
multiTracker = cv2.MultiTracker_create()
 
# Initialize MultiTracker 
for bbox in bboxes:
  multiTracker.add(createTracker(), first_f, bbox)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # get updated location of objects in subsequent frames
    success, boxes = multiTracker.update(frame)

    # draw tracked objects
    for i, newbox in enumerate(boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

    # show frame
    cv2.imshow('MultiTracker', frame)

    # quit on q button
    if cv2.waitKey(10) & 0xFF == ord('q'):  # q pressed
        break

