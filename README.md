# pairwise-combo-cancel-rootmotion

## 领域

游戏、图形与多媒体

## 主题

招式取消与根运动

## 说明

事件按时间排序、增量回滚与单层写入。

初始状态下 `gameplay/combo_cancel_rootmotion.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
