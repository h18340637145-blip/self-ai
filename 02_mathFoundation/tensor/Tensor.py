
from functools import reduce


class Tensor:
    def __init__(self, data, shape=None):
        if isinstance(data, (list, tuple)):
            self._data, self._shape = self._flatten_nested(data)
        elif isinstance(data, np.ndarray):
            self._data = data.flatten().tolist()
            self._shape = tuple(data.shape)
        else:
            self._data = [data]
            self._shape = ()

        if shape is not None:
            total = reduce(lambda a, b: a * b, shape, 1)
            if total != len(self._data):
                raise ValueError("Provided shape does not match the number of elements in data.")
            self._shape = tuple(shape)
    def _flatten_nested(self, data):
        if not isinstance(data, (list, tuple)):
            return [data], ()
        flat_data = []
        shape = []
        for item in data:
            sub_flat, sub_shape = self._flatten_nested(item)
            flat_data.extend(sub_flat)
            if not shape:
                shape = list(sub_shape)
            elif shape != list(sub_shape):
                raise ValueError("Inconsistent shapes in nested data.")
        return flat_data, (len(data), *shape)
    @staticmethod
    # 计算给定形状的步长（strides）
    def _compute_strides(shape):
        if(len(shape) == 0):
            return ()
        strides = [1] * len(shape) # 最后一个维度步长一定是 1
        for i in range(len(shape) - 2, -1, -1): # （3，4）-> shape=2  range(0, -1, -1) 即仅 0 参与循环
            strides[i] = strides[i + 1] * shape[i + 1] # strides[0] = strides[1] * shape[1] = 1 * 4 = 4  所以步长就是(4, 1)  最外侧维度移动一次，步长是 4，相当于移动 4 个格子，内层维度移动一次，步长是 1，相当于移动 1 个格子。
        return tuple(strides)
    # 对于形状(3, 4)，步长为(4, 1)——跳过 4 个元素前进一行，跳过 1 个元素前进一列

# 重塑，挤压，放松
t = Tensor(list(range(12)), shape=(2,6))
print(f"t={t}")
# r = t.reshape((3,4))
# r = t.reshape((-1, 3))

# 转置和置换
# mat = Tensor(list(range(6)), shape=(2,3))
# print(f"mat={mat}")
# 转置操作会交换两个坐标轴
# tr = mat.transpose(0, 1)
# t4d = Tensor(list(range(24)), shape=(1,2,3,4))
# 重排操作会重新排列所有坐标轴
# perm = t4d.permute(0,2,3,1)
# print(f"perm={perm}")