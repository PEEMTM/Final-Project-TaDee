import ultralytics
ultralytics.checks()

import torch
print(torch.__version__)
print(torch.cuda.is_available())