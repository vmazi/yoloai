from ultralytics import YOLO
import json

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    source = './datasets/coco128/images/train2017/000000000142.jpg'
    results = model.predict(source, show=True)  # list of Results objects
    results_json = json.loads(results[0].tojson())
    for result in results_json:
        print(result['name'])
