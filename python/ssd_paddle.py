import argparse
import numpy as np
import cv2
from paddle.fluid.core import PaddleBuf
from paddle.fluid.core import PaddleDType
from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor

models_dir = "../models/"
modelname = "paddle-mobilenet-ssd"
if modelname == "mobilenet-ssd":
    input_size = 300
else:
    input_size = 160

def clip_bbox(bbox):
    xmin = max(min(bbox[0], 1.), 0.)
    ymin = max(min(bbox[1], 1.), 0.)
    xmax = max(min(bbox[2], 1.), 0.)
    ymax = max(min(bbox[3], 1.), 0.)
    return xmin, ymin, xmax, ymax

def preprocess(img):
    img = cv2.resize(img, (input_size, input_size))
    img = img.transpose((2,0,1))
    if modelname == "mobilenet-ssd":
        img = (img - 127.5) * 0.007843
    else:
        mean = np.array([103.94, 116.669, 123.68], np.float32).reshape([3, 1, 1])
        img = img - mean
    image = PaddleTensor()
    image.name = "data"
    image.shape = [1, 3, input_size, input_size]
    image.dtype = PaddleDType.FLOAT32
    image.data = PaddleBuf(img.flatten().astype("float32").tolist())
    return [image]
    
def draw_result(img, out):
    h, w, _ = img.shape
    for dt in out:
        if len(dt) < 5 or dt[1] < 0.5:
            continue
        xmin, ymin, xmax, ymax = clip_bbox(dt[2:])
        xmin=(int)(xmin*w)
        ymin=int(ymin*h)
        xmax=(int)(xmax*w)
        ymax=int(ymax*h)
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255))
        if ymin<20:
            ymin=20
        cv2.putText(img,"detected",(xmin,ymin),3,1,(255,0,0))
    cv2.imshow("img",img)
    cv2.waitKey(1)

def test_image(predictor, imgpath):
    img = cv2.imread(imgpath)
    inputs = preprocess(img)
    outputs = predictor.run(inputs)
    output = outputs[0].as_ndarray()
    draw_result(img, output)
    cv2.waitKey()

def test_camera(predictor):
    cap = cv2.VideoCapture("../images/test.avi")
    while True:
        ret, img = cap.read()
        if not ret:
            break
        inputs = preprocess(img)
        outputs = predictor.run(inputs)
        output = outputs[0].as_ndarray()
        draw_result(img, output)

def main():
    args = parse_args()
    model_file = args.model_dir + "/__model__"
    params_file = args.model_dir + "/params"
    config = AnalysisConfig(model_file, params_file)
    config.disable_gpu()
    predictor = create_paddle_predictor(config)
    test_image(predictor, args.image_path)
    #test_camera(predictor)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=models_dir+modelname, help="program filename")
    parser.add_argument("--image_path", default="../images/000001.jpg", help="image path")
    return parser.parse_args()

if __name__ == "__main__":
    main()