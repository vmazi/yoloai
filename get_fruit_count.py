from ultralytics import YOLO
import json
import collections
from itertools import chain

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    source = './datasets/coco128/images/train2017/000000000142.jpg'
    results = model.predict(source, stream=True)  # list of Results objects
    results_json = map(lambda res: json.loads(res.tojson()), results)
    item_names_by_result = map(
        lambda inference_list: collections.Counter(map(lambda name: name['name'], inference_list)), results_json)
    print(list(item_names_by_result))
