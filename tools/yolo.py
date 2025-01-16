import numpy as np
import torch
import torchvision.ops as ops

from tools.utils import (load_onnx_model, scale_coords, letterbox,
                         check_img_size)


class YOLO:
    def __init__(self, model_path, conf_thres=0.5, nms_thres=0.45,
                 img_size=640, stride=32):
        # Load model
        self.session = load_onnx_model(model_path)
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres

        self.img_size = img_size
        self.stride = stride

        # check if img_size can be divided by stride or not
        self.imgsz = check_img_size(img_size, s=self.stride)

        self.mean = 0.0
        self.scale = 0.00392156862745098

    def model_inference(self, input):
        input_name = self.session.get_inputs()[0].name
        output = self.session.run([], {input_name: input})
        return output

    def inference(self, img0):
        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=False)[0]

        # Convert
        img = img[:, :, ::-1]  # BGR to RGB
        img = np.ascontiguousarray(img)

        img = (img - self.mean) * self.scale
        img = np.asarray(img, dtype=np.float32)
        img = np.expand_dims(img, 0)
        img = img.transpose(0, 3, 1, 2)

        output = self.model_inference(img)
        return self.post_process(img, img0, output[0])

    def xywh2xyxy(self, x):
        y = x.copy()
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def post_process(self, img, img0, output):
        """
        Draw bounding boxes on the input image. Dump boxes in a txt file.
        """
        output = output.squeeze(0).T

        det_bboxes = self.xywh2xyxy(output[:, :4])
        det_scores = output[:, 4]
        det_kpts = output[:, 5:]

        # Rescale boxes from img_size to img0 size
        scale_coords(img.shape[2:], det_bboxes, img0.shape, kpt_label=False)
        scale_coords(img.shape[2:], det_kpts, img0.shape, kpt_label=True,
                     step=3)

        keep = det_scores > self.conf_thres
        det_bboxes = det_bboxes[keep]
        det_scores = det_scores[keep]
        det_kpts = det_kpts[keep]

        if len(det_bboxes):
            keep = ops.nms(torch.tensor(det_bboxes),
                           torch.tensor(det_scores),
                           self.nms_thres).numpy()

            return det_bboxes[keep, :], det_kpts[keep, :]

        return np.array([]), np.array([])
