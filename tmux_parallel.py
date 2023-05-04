import os

STORE = "__store"
NOT_STORE = "__not_store"

def set_up_tmux(gpus=[2,3,4,5,6,7]):
    '''也可以手动提前建好，默认tmux名为gpus2、gpus3等
    '''
    pass

def add_tmux_affix(commands, gpus=[2,3,4,5,6,7], conda_env = 'CG-ODE'):
    '''将commands平均分配到各个tmux终端（对应gpu）上运行
    '''
    output = []
    num_gpu = len(gpus)
    for i, com in enumerate(commands):
        new_com = f"conda activate {conda_env} \n CUDA_VISIBLE_DEVICES={gpus[i % num_gpu]} {com}" 
        new_com = "tmux send-keys -t gpu%d \"%s\" C-m;"% (gpus[i % num_gpu], new_com)
        output.append(new_com)
    return output

__dic = {}
def get_multiplied(chois_params:dict[str, list]):
    '''自动叉积
    输入字典: key是参数名, value是该参数可选值构成的list
    params_list: 各参数可选值交叉组合构成的所有参数取值字典
    commands: 根据参数取值生成的命令字符串
    '''
    assert all([isinstance(v, list) or isinstance(v, range)
                 for v in chois_params.values()]), "迭代的必须是list或range"
    global __dic
    __dic = chois_params # 或许是eval的某种问题，不接这个没用的赋值就会报错
    params_list = eval("[{"+ "".join(["'"+k+"':"+k+", " for k in __dic.keys()]) +"}"+ "".join(["for "+k+" in " "__dic['"+k+"'] " for k in __dic.keys()]) +']')
    return params_list

def param2command(python_file, params:dict):
    "将参数字典转化为命令"
    param_list = ["--%s"%k if v == STORE else
                  "" if v == NOT_STORE else
                  "--%s=%s"%(k,v) for k,v in params.items()]
    command = f"python {python_file} " + " ".join(param_list)
    return command

# for com in commands:
#     os.system(com)