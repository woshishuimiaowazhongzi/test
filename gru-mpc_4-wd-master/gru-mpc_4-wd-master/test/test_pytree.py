import torch
from torch.utils._pytree import tree_flatten

# 一个嵌套数据结构
example_tree = {
    'tensor': torch.tensor([1, 2]),
    'nested_list': [10, 20],
    'number': 5
}

# 定义一个函数，告诉 tree_flatten 将所有列表视为叶子节点
def my_is_leaf(node):
    # 如果节点是一个列表，就将其标记为叶子，不再深入展开
    return isinstance(node, list)

# 使用自定义的 is_leaf 函数进行展开
leaves_custom, treespec_custom = tree_flatten(example_tree, is_leaf=my_is_leaf)
print("自定义后的叶子节点:", leaves_custom)
# 输出: [tensor([1, 2]), [10, 20], 5]