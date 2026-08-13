from tensorflow import reduce_mean, reduce_max
from keras.layers import Concatenate, Multiply
from keras.layers import Conv1D, Layer


class AttributeAttention(Layer):
    def __init__(self, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = Conv1D(filters=1,
                           kernel_size=kernel_size,
                           padding='same',
                           activation='sigmoid')

        self.concatenate = Concatenate()
        self.multiply = Multiply()

    def call(self, main_branch):
        # Mean pooling
        x1 = reduce_mean(main_branch, axis=-1, keepdims=True)

        # Max pooling
        x2 = reduce_max(main_branch, axis=-1, keepdims=True)

        # Concatenate
        out = self.concatenate([x1, x2])

        # Attention mask
        out = self.conv(out)
        out = self.multiply([main_branch, out])
        # Apply attention
        return out


# def AttributeAttention(main_branch, kernel_size=3):  # main_branch  = input feature map
#
#     # Average Pooling
#     x1 = reduce_mean(main_branch, axis=1)
#     x1 = expand_dims(x1, axis=1)
#
#     # Max Pooling
#     x2 = reduce_max(main_branch, axis=1)
#     x2 = expand_dims(x2, axis=1)
#
#     # Concatenate
#     out = Concatenate()([x1, x2])
#
#     # Conv layer
#     out = Conv1D(1, kernel_size=kernel_size, padding='same', activation='sigmoid')(out)
#     out = Multiply()([main_branch, out])
#
#     return out
