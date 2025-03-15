import os
import random
import numpy as np
import tensorflow as tf

def set_tensorflow_seed(seed = 0, determinism = True):
    """
    https://github.com/Burf/TFDetection/blob/main/tfdet/core/util/random.py
    # This is the random seed initialization code that has to be at the beginning.
    """
    if determinism:
        tf_version = float(".".join(tf.__version__.split(".")[:2]))
        if hasattr(tf.config.experimental, "enable_op_determinism"):
            tf.config.experimental.enable_op_determinism()
        else:
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            if tf_version == 2.9:
                os.environ["TF_DISABLE_DEPTHWISE_CONV_DETERMINISM_EXCEPTIONS"] = "1"
            if 0 < len(get_device("gpu")):
                os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
                try:
                    from tfdeterminism import patch
                    patch()
                except:
                    print("Please install 'tensorflow-determinism', and it will be more specific.")
        
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_DISABLE_DEPTHWISE_CONV_DETERMINISM_EXCEPTIONS"] = "1"
        
    tf.random.set_seed(seed)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    
    
def set_seed(seed = 0, determinism = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    set_tensorflow_seed(seed, determinism=True)