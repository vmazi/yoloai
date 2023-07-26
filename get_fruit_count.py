from ultralytics import YOLO
import json
from collections import Counter


def get_item_counts(model_to_use, source_to_use):
    results = model_to_use.predict(source_to_use, stream=True)  # list of Results objects
    results_json = map(lambda res: json.loads(res.tojson()), results)
    return map(lambda inference_list:
               Counter(map(lambda inference:
                           inference['name'], inference_list)),
               results_json)


if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    source = './datasets/coco128/images/train2017/000000000142.jpg'

    print(get_item_counts(model, source).__next__())
# results.stream()
#        .map(Result::toJson)
#        .map(array -> array.stream()
#                           .collect(
#                                   Collectors.groupingBy(
#                                           Inference::getName, Collectors.counting()))
