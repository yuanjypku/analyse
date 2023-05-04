# LoadLog: tensorboard分析分析工具（未来或许会写成模块）

1. 单个实验记录的提取：包括训练过程的scalar和实验的hparam与metric（默认每次实验只记录训一次，保存在实验文件夹内的.0及其子文件中的.1内）
2. 一组实验记录的统一提取：统一文件夹下的多个上述单个实验记录的合集。
3. 为上述实验记录提供一些分析方法
4. 尽可能保留原有的tensorboard模块

# tmux_parallel: Tmux 并行跑实验

用于自动同时跑多组实验
借用tmux在命令行中实现并行不同参数。好处是即插即用，不用重新包装成async函数来实现并行，原来的命令行调用方法接着用