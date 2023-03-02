# Install
## Installation guide : https://towardsdatascience.com/the-ultimate-tensorflow-gpu-installation-guide-for-2022-and-beyond-27a88f5e6c6e




#  https://developer.nvidia.com/cuda-downloads
# 
# https://developer.nvidia.com/rdp/cudnn-download
#
# https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
#  Follow installation guide
# https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

print("")