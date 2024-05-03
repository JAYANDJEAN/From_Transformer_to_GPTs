import torch

# Check that MPS is available
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    print(mps_device)
else:
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

