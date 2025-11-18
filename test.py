import torch
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Testing Numpy")
    a=np.random.rand(3, 3)
    print(a)

    print("\n Testing Pytorch")
    x=torch.randn(3, 3)
    print(x)

    print("\n Checking if CUDA is available")
    if torch.cuda.is_available():
        print("GPU name: ", torch.cuda.get_device_name(0))
        x=x.to("cuda")
        print("Tensor moved to GPU: ", x.device)
    
    print("\n Testing Matplotlib")
    plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
    plt.title("Matplotlib Test")
    plt.savefig("matplotlib_test.png")
    
    print("All tests passed")
if __name__=="__main__":
    main()