import gc
import sys
import torch
import subprocess as sp
from collections import defaultdict
import operator
from functools import reduce

GPU = "cuda"
CPU = "cpu"

        
def prod(iterable):
    return reduce(operator.mul, iterable, 1)

def get_size_to_live_tensors(device=None):
    if device == "GPU":
        device = "cuda"
    if device == "CPU":
        device = "cpu"
    if device not in (None, CPU, GPU):
        raise ValueError("Invalid device")

    shape_elsize_to_count = defaultdict(int)
        
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and ((device is None) or (obj.device.type == device)):
                # or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                shape_elsize_to_count[(tuple(obj.size()), obj.element_size())] += 1
        except:
            pass
    
    to_return = {}
    for (shape, elsize), count in shape_elsize_to_count.items():
        total_size = prod(shape) * elsize * count / 1e9
        to_return[total_size] = (shape, count)

    return dict(sorted(to_return.items(), key=operator.itemgetter(0), reverse=True))

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def show_locals(dict_):
    """ E.g. dict_ can be locals() or globals() """
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in dict_.items()), 
                             key= lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
        
        
def get_tensor_size(tensor):
    return tensor.nelement() * tensor.element_size()


def print_gpu_mem_usage():
    free, total = torch.cuda.mem_get_info()
    print(f"{(total - free) / 1e9:.2f} / {total / 1e9:.2f} GB used ({100 * (total - free) / total:.1f}%). {free/1e9:.2f} GB free")

# torch.cuda.mem_get_info
# torch.cuda.memory_summary()
# torch.cuda.memory_allocated
# GPUtil.showUtilization()


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for x in memory_free_info]
    if len(memory_free_values) > 1:
        import pdb; pdb.set_trace()
    return memory_free_values[0]