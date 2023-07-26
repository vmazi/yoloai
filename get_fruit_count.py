from ultralytics import YOLO
import json
import collections

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    source = './datasets/coco128/images/train2017/000000000142.jpg'
    results = model.predict(source)  # list of Results objects
    results_json = json.loads(results[0].tojson())
    item_names = map(lambda json_res: json_res['name'], results_json)
    freq = collections.Counter(item_names)

    for (key, value) in freq.items():
        print(key, " -> ", value)
