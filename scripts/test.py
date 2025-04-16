import torch
print(torch.cuda.is_available())     # 应该输出 True
print(torch.version.cuda)            # 应该输出 11.8
print(torch.cuda.get_device_name(0))