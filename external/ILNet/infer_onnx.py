import cv2 
import numpy as np
import onnxruntime as ort



model = ort.InferenceSession('LW_Net.onnx', providers=['CUDAExecutionProvider'])

img_path = '/mnt/data/danhnt/Seg/ILNet-master/datasets/IRSTD-1k/images/Misc_1.png'

img = cv2.imread(img_path)

img = cv2.resize(img, (640, 512))

img = img.transpose(2, 0, 1)

img = img[np.newaxis, ...]

img = img.astype(np.float32)

img = img / 255.0

img = (img - 0.5) / 0.5

img = img.astype(np.float32)

img = np.ascontiguousarray(img)

input_name = model.get_inputs()[0].name

output_name = model.get_outputs()[0].name

result = model.run([output_name], {input_name: img})

result = result[0]

result = result > 0.5

result = result.astype(np.uint8)

result = result[0, 0, :, :]
result = result * 255

cv2.imwrite('result_onx.png', result)

print('Done')
