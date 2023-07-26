from ultralytics import YOLO
import json
import collections
from itertools import chain
if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    source = './datasets/coco128/images/train2017/000000000142.jpg'
    results = model.predict(source, stream=True)  # list of Results objects
    results_json = map(lambda res: json.loads(res.tojson()), results)
    items = chain.from_iterable(results_json)
    item_names = map(lambda json_item: json_item['name'], items)
    freq = collections.Counter(item_names)

    for (key, value) in freq.items():
        print(key, " -> ", value)
3
