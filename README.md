# StackConvPrun
DenseNet &amp; Resnet as Backbone to Perform Lahan Gambut Classification with Self-Attention Pruning Mechanism

# Initial Research Problem

The utilization of a concatenation layer based on two backbone architectures, such as DenseNet and ResNet, introduces additional computational overhead during both training and inference. In this study, we propose the implementation of pruning based on a self-attention mechanism scoring. This pruning mechanism aims to selectively remove layers in the backbone/head section, reducing computational requirements. The decision to prune a layer is determined by comparing its attention mechanism score to the standard deviation of the total attention mechanism scores.

*note: This approach introduces variability in the pruned layers across different use cases, and we address this by initializing random variables for each layer.

# Conclusion

Based on the evaluation results, we successfully reduced the training time of the baseline model from 30 minutes per epoch to just 1 minute. While the evaluation results show a trade-off, the achieved improvements in computational efficiency are deemed acceptable.a

# Evaluation Result

## Before Pruning
![](images/Evaluation%20Metrics%20Before%20Pruning%20Attention.png)

## After Pruning
![](images/Evaluation%20Metrics%20After%20Pruning%20Attention.png)