# 稿件同步整理说明

当前配对稿见 `output/pdf/manuscript_synchronized_review_v7/README.md`。
旧Fig.1误插来自v6生成代码硬编码v3；后来人工批准的engine-v4在版本authority中已经存在。
本轮通过既有LangGraph的新profile重新绑定资产，同时从同一次干净构建产生正文/SM。

重要方法澄清：Stage I/II的运行profile是full；Stage III是topo_trivial，Sobol初始化；Stage IV采用三输入96单元MLP及独立的提议池/通道配额选择器。不能把代码中的可选响应清理分支当成这些端点实际运行。响应量仍保留为训练目标。

原自由能表达及讨论未改，未计算误差或证明可忽略。连续闭隙与全局极小值跳变的区别、单-q范围、uniform-SC拓扑掩膜等问题已列作者确认表，不由排版任务作科学裁决。

后续报告应继续区分选择记录、唯一请求坐标、留存坐标及总执行尝试；本包没有完整执行账本，不能将位置比例当作实测时间加速。
