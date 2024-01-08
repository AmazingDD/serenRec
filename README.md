# SR-SNN
This is the repository of our article published in xxxx 2025 "SR-SNN: When Spiking Neural Network Meets On-device Session-based Recommendation"

## 思路

因为现在用SNN和推荐系统相关的研究可以说是极其稀少，仅有的一些相关研究也和我们要用的LIF关系不大，例如24年elsvier的那个，还有中山大学22，23年发的（主要是在GNN基础上用SNN进行特征提取与后续的建模）；

由于SNN可以理解成是RNN的一种变体，而RNN在推荐领域的其中一个成熟应用场景是序列推荐（Sequential Recommendation），当然这里面可能根据序列长短，用户匿名等特点又抽出了一种叫会话推荐（session recommendation）的场景，但本质仍是一种序列。近几年来推荐系统的相关顶会（CCF-A）文章可谓是五花八门，但近三年内围绕序列推荐的类似研究仍然会被顶会接收（截止23年底），所以我们可以认为这种序列推荐场景依旧可以在未来一到两年内作为我们的主要应用方向进行探索

再更细化下场景，可以做的是序列推荐中的next-item recommendation

从推荐算法的角度讲，现有推荐算法如果在不引入其他附加信息（side information）的情况下，可以分为以下几种：

 - 传统方法：如Pop等这种基于一定规则的筛选推荐，虽然他们没法被SNN化，但经常会被拿出来和其他sota进行对比；还有一些如（BPR）MF，FPMC这种基于矩阵分解的方式，这类方法在序列场景下是会被针对性的修改成Seq/session-BPRMF的形式，因此我们同样可以将其SNN化，也即Seq/session-SNN-BPRMF这样的思路
 - RNN-base的方法：比如stamp，narm，sasrec, bert4rec这类sota，他们之中或多或少都会在原本LSTM/GRU的基础上再加一些如attention，transformer等机制优化，ijcai23有一篇文章发出来就是设计一种新的attention方式换掉这些方法中的特定部分，成功发表了文章，因此我在想我们也有spikeformer和spikeattention这种参考方法，能不能也做一些这种修改；或者是像GRU4rec这种特别直白的应用，我们能不能直接弄个SNN4rec？
 - GNN-based 中山大学的团队有逐步尝试往SNN转的思路，我们可能会和他们的思路冲突，所以这类方法的探究可能优先级不高，但也要考虑
 
 以上这些思路单独拿出来写文章肯定是工作量不太够的，所以我想综合起来做一个完整的研究，不知道是否可行

可以预期的是SNN化肯定准确率大打折扣，所以赛道可能还是要集中在损失相对小的准确率的基础上耗能显著减少这里，那具体的耗能统计这次恐怕不能单纯的算脉冲这么解决了

现在的一切都是理论，因为真的没人做这方面的事情，所以这些工作真的需要人一点点的做起来验证结果，可能工作量很大


## TODO list

 - [ ] 一些计划
 - [ ] 一些计划