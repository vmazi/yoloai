from ultralytics import YOLO
import json
from collections import Counter


def get_item_counts(model_to_use, source_to_use, names):
    results = model_to_use.predict(source_to_use, stream=False, verbose=True, device=0, show=False)
    return map(lambda result:
               Counter(map(lambda box: names[int(box.cls)],
                           result.boxes)),
               results)


if __name__ == '__main__':
    model = YOLO('./runs/detect/train22/weights/best.pt')
    source = './datasets/Cucumber-Slice-Counter-1/test/images/newcucumber_0_331_jpeg_jpg.rf' \
             '.c68c395fde2e563f58d617b7b7c41910.jpg'
    names = model.names
    for item_count in get_item_counts(model, source, names):
        print(item_count)
# results.stream()
#        .map(Result::toJson)
#        .map(array -> array.stream()
#                           .collect(
#                                   Collectors.groupingBy(
#                                           Inference::getName, Collectors.counting()))
