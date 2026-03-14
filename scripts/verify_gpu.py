# gpu verification

import torch


def main():
    print("=" * 50)
    print("GPU Verification")
    print("=" * 50)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version:    {torch.version.cuda}")
        print(f"GPU count:       {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  VRAM:          {props.total_mem / 1e9:.1f} GB")
            print(f"  Compute cap:   {props.major}.{props.minor}")
            print(f"  SM count:      {props.multi_processor_count}")

        print(f"\nbf16 support:    {torch.cuda.is_bf16_supported()}")

        print("\nRunning quick matmul benchmark...")
        x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        import time
        t0 = time.time()
        for _ in range(100):
            y = x @ x
        torch.cuda.synchronize()
        dt = time.time() - t0
        tflops = 100 * 2 * 4096**3 / dt / 1e12
        print(f"  bf16 matmul:   {tflops:.1f} TFLOPS")
    else:
        print("\nNo CUDA GPU detected. Training will be very slow on CPU.")
        print("Make sure you have installed the CUDA version of PyTorch.")
        print("See: https://pytorch.org/get-started/locally/")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
