# LossBalance 合并说明

v2只改变ROI重建损失：按阳性样本频率N_c^(-alpha)加权，alpha=.5、ratio cap4，
归一化使期望前景权重质量为1；lambda_roi=1。背景不参与，batch分母固定。
v3a仅优化present且kept器官的positive ROI，lambda_positive=.25；removed仅监控，
仍受global L2+LPIPS监督。保持原OWT架构与mask日程；不加PSEM或Delta。
这两者不是OrganSlot的Tversky/BalancedFocal分割损失。

WORD224控制配置：20 tokens/class、有效batch64、blr1e-4、1200epochs。
记录指标必须包括Direct/Indirect raw/post、precision、recall、volume ratio、重建误差。
v2源记录的训练1586123完成，3D full1586124启动前取消；v3源记录停留旧pending，
不能当作今天状态。本分支原docs不含完整正式八类结果，不凭其他协议补排名。
跨版本已汇总结果见experiment/psem分支RESULTS；需要汇报时回到对应原始结果核验。

公式推导、样本频率/权重表、单元测试和历史任务完整保存在
[历史原文](archive/HISTORY.md)。不因归档删除数据、权重或实验可追溯性。
