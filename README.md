# MSA
记录第二个项目从数据加载到模型文件修改的过程

原始多模态数据（视频 / 音频 / 文本 / 标注）
        ↓
读取标注文件（情感标签）
        ↓
对齐：音频 ↔ 文本 ↔ 情感
        ↓
过滤异常样本（太短、没标签）
        ↓
统一成 (audio_path, transcript, label_id)
        ↓
shuffle
        ↓
train / val / test 划分
        ↓
pickle 保存
