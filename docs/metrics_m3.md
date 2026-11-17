# M3 多视角融合评测记录

> 计划验收：同场景重复 10+ 次，记录吸点 / 法向重复性，对比单帧。

## 配置

| 项 | 值 |
|----|----|
| `multiview_n` | 5 |
| `multiview_radius_m` | 0.03 |
| `multiview_voxel_m` | 0.004 |
| 场景 | Mock / 真机（填写） |

## 指标表（模板）

| 轮次 | 模式 | c*_x | c*_y | c*_z | n 角相对均值 ° | 成功视点 | 备注 |
|------|------|------|------|------|----------------|----------|------|
| 1 | multiview | | | | | | |
| 2 | multiview | | | | | | |
| … | | | | | | | |
| 汇总 σ | | | | | | | |

单帧对照：对同一批抓帧只跑 `fine_localize_single`，填 σ 对比。

## 命令

```bash
# Mock smoke（需 .venv312）
./start_web.sh   # 或 API
# POST /api/pipeline/coarse → /api/pipeline/move_safe → /api/pipeline/multiview
```

## 结论

（填写：融合相对单帧是否改善、失败模式）
