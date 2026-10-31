import torch
print('torch', torch.__version__, '| cuda build', torch.version.cuda)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print('GPU:', p.name, '| sm_%d%d' % torch.cuda.get_device_capability(0), '| %.1f GiB' % (p.total_memory/1024**3))