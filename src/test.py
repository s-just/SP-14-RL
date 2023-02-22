import os
from ultralytics import YOLO
import object_recognition as OR


def extract_objects(tensor):
    # Loops through each detected object stored by the output tensor and returns them as a list
    objects = [obj for obj in tensor]  # This single line is put into a function for my own readability.
    return objects


def get_overlaps(objects):
    found_overlaps = []  # This could be declared outside the scope of this function to check specific overlaps during runtime.

    # Loop through each object that hasn't been compared yet and check for overlap with other objects
    compared = set()
    for i, obj1 in enumerate(objects):
        for j, obj2 in enumerate(objects):
            if i >= j:
                continue
            if (i, j) in compared:
                continue
            # Check if the bounding boxes intersect
            x1_min, y1_min = obj1[0], obj1[1]
            x1_max, y1_max = obj1[2], obj1[3]
            x2_min, y2_min = obj2[0], obj2[1]
            x2_max, y2_max = obj2[2], obj2[3]
            if (x1_min <= x2_max and x1_max >= x2_min and
                    y1_min <= y2_max and y1_max >= y2_min):
                found_overlaps.append((i, j))
                found_overlaps.append((i, j))
                found_overlaps.append((j, i))
    # Return whether any overlaps have been found.
    if len(found_overlaps) > 0:
        return True
    else:
        return False


# Set current working directory and load model
HOME = os.getcwd()
model = YOLO(f'{HOME}/src/weights/best.pt')

while True:
    # Test object recognition by getting data from screen
    obj_recognition = OR.ObjectRecognition(model, (0, 40, 640, 480), True, 0.2)
    result = obj_recognition.get_screen_data()
    print(result)

# Test overlap detection
# all_objects = [[-3, -3, -2, -2], [2, 2, 4, 5]]
# overlaps = get_overlaps(all_objects)
# print("Test Case 1: ([-3, -3, -2, -2], [2, 2, 4, 5])", " - Result : ", overlaps)
#
# all_objects = [[3, 3, 2, 2], [2, 2, 4, 5]]
# overlaps = get_overlaps(all_objects)
# print("Test Case 2: ([3, 3, 2, 2], [2, 2, 4, 5])", " - Result : ", overlaps)
#
# all_objects = [[2, 2, 4, 5], [3, 3, 2, 2]]
# overlaps = get_overlaps(all_objects)
# print("Test Case 3: ([2, 2, 4, 5], [3, 3, 2, 2])", " - Result : ", overlaps)

# Currently we can't test with the Object Recognition model as we need to finish labeling the data, so it can be
# trained and loaded in Python.
