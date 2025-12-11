- 无 bias 的 matmul，基本上是按照 MatmulCustomMultiCore 的 sample 代码（带 bias 玩的版本）改的
- 源代码在 ./MatmulCustom 下
- 相应的改了 Acl 的代码和脚本

```
C = A * B
```

soc_version = 910B2

1. 编译
```
bash install.sh
```

如果 msopgen 命令报错类似如下
```
2025-09-29 21:42:30 (1031) - [ERROR] The path CooSpmmCustom.json should not be written by user group or others, which will cause security risks
```
请执行
```
chmod -R go-w .
```
将当前目录下的所有文件/目录的非用户写权限去除，即可正常运行。

2. 部署

执行生成的 Op 目录下的 build_out 下的 .run 文件
（第一步执行完最后会输出该 run 文件的路径）

3. 调用 & 测试
```
cd AclNNInvocation/
bash run.sh
```
