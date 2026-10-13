# Instructor interpretation of the recorded CPU experiment

The declared rule favors learning rate 0.04 only if every candidate-minus-baseline final validation difference is negative. The observed differences are approximately -0.05832, -0.04929 and -0.04662 for initialization seeds 11, 23 and 37. All satisfy that rule. Their mean is -0.05141 and their sample SD is 0.00613.

For seed 11, baseline validation MSE is about 0.14520 and candidate MSE is 0.08689. The subtraction is negative because the candidate is lower. Both are below the training-mean constant predictor's validation MSE of 0.81955. That checks that the learned models improve on a meaningful simple baseline, while the paired comparison asks the narrower question about learning rate.

The candidate mean falls faster and finishes lower in both curve panels. The training panel and validation panel summarize different examples. Their agreement supports the small fixture result; it does not eliminate overfitting risk or establish performance on new datasets. The shaded range describes only the three initialization seeds, and the SD is neither a confidence interval nor variation over data splits.

I would retain the candidate for this declared 40-update fixture experiment. Before generalizing, I would declare a second study using additional initialization seeds and an independently generated data split, while keeping the architecture, metric and update budget fixed. If selection continued after inspecting those results, a separate untouched final evaluation would be needed. This report contains validation-based selection only.

The intentionally failed first attempt of baseline-s11 remains in the suite. Its retry starts from initialization and reproduces the same recipe; it is not numerical checkpoint resume. CPU completion does not establish GPU/TPU numerical parity or performance. Optional tracker adapters were checked in a separate small training run and did not generate the figures shown here.
