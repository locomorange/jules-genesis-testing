import sys

print("Attempting to import dependencies as specified in pyproject.toml...")

try:
    print("Importing genesis (from genesis-world)...")
    import genesis
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

    print("All specified dependencies imported successfully.")
    sys.exit(0)

except ImportError as e:
    print(f"Failed to import one or more dependencies: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}", file=sys.stderr)
    sys.exit(1)
