import torch

# 加载两个检查点
checkpoint1 = torch.load('/userhome/lyd/Pointcept/exp/model_best_frozen_dcmodel_66_38.pth')
checkpoint2 = torch.load('/userhome/lyd/Pointcept/exp/scanrefer/debug/model/model_last_6553.pth')

# 获取参数字典
state_dict1 = checkpoint1['state_dict']
state_dict2 = checkpoint2['state_dict']

# state_dict1 = {}
# for key, value in state_dict1_.items():
#     if "module." in key:
#         state_dict1[key[7:]] = state_dict1_[key]

# 比较参数
for name, param in state_dict1.items():
    if name in state_dict2:
        if ((param == state_dict2[name]).all()).item() == False:
            print("====== Different parameters: " + name)
        else:
            print("====== Same parameters: " + name)
        # print(f"Parameter {name} is present in the second checkpoint.")
    else:
        print(f"Parameter {name} is not present in the second checkpoint.")
