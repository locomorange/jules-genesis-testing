import sys

try:
    print("Importing genesis (as genesis_world)...")
    import genesis  # MODIFIED HERE
    print("Successfully imported genesis")

    print("Importing gymnasium...")
    import gymnasium
    print("Successfully imported gymnasium")

    print("Importing torch...")
    import torch
    print("Successfully imported torch")

    print("Importing torchaudio...")
    import torchaudio
    print("Successfully imported torchaudio")

    print("Importing torchvision...")
    import torchvision
    print("Successfully imported torchvision")

    print("All main dependencies imported successfully.")
    sys.exit(0)

except ImportError as e:
    print(f"Failed to import one or more dependencies: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred: {e}", file=sys.stderr)
    sys.exit(1)
