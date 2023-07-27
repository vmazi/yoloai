from ultralytics import YOLO
from collections import Counter


def get_item_counts(model_to_use, source_to_use, names):
    results = model_to_use.predict(source_to_use, stream=False, verbose=True, device=0, show=False)
    return map(lambda result:
               Counter(map(lambda box: names[int(box.cls)],
                           result.boxes)),
               results)


if __name__ == '__main__':
    model = YOLO('./runs/detect/train8/weights/best.pt')
    source = './datasets/Cucumber-Slice-Counter-1/train/images/newcucumber_0_455_jpeg_jpg.' \
             'rf.bf8a8ea7d2eba2a905efefc625653c81.jpg'
    names = model.names
    for item_count in get_item_counts(model, source, names):
        print(item_count)
# results.stream()
#        .map(Result::toJson)
#        .map(array -> array.stream()
#                           .collect(
#                                   Collectors.groupingBy(
#                                           Inference::getName, Collectors.counting()))
