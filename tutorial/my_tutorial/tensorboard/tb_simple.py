import os
import sys
from torch.utils.tensorboard import SummaryWriter

file_path = os.path.join(sys.path[0])
print(file_path)

if __name__ == '__main__':
    writer = SummaryWriter(file_path + '/log/scalar_example')
    for i in range(10):
        writer.add_scalar('quadratic', i**2, global_step=i)
        writer.add_scalar('exponential', 2**i, global_step=i)

    writer.close()

