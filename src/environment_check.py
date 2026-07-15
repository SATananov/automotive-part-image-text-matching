import sys

import keras
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"TensorFlow: {tf.__version__}")
    print(f"Keras: {keras.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"pandas: {pd.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"Available devices: {tf.config.list_physical_devices()}")


if __name__ == "__main__":
    main()
