# TP3 · Exercise 2 · Speaker script

One section per slide of `TP3_exercise2.pptx` (the same text is in each slide's speaker notes). Roughly 1–1.5 minutes per slide.

## Slide 1 · 2: OUR MULTILAYER PERCEPTRON (Built from scratch with numpy, fully vectorized)

This is the model we use for the whole exercise: a multilayer perceptron written from scratch with numpy.

On the left you see the idea. Each image is 28 by 28 pixels, which we flatten into 784 inputs. Those go through one hidden layer, here 128 sigmoid units, and then into 10 output units, one per digit. The predicted digit is simply the output unit with the highest value.

Every layer is a weight matrix. The forward pass is h = W times V plus b, followed by the activation. Backpropagation sends the error back with W transposed times delta, multiplied by the derivative of the activation.

We put the samples of a batch in the columns of the input matrix, so a whole batch goes through a layer in one matrix product. That is why a 60-epoch run takes seconds instead of minutes.

To make sure the implementation is right we wrote tests: we check backpropagation against a numerical derivative, against one step we calculated by hand for the 2-2-1 network, and we check that both 2-2-1 and 2-3-2-1 learn XOR.

## Slide 2 · 2A: HOW DO WE EVALUATE OUR SYSTEM?

Question 2a: how do we evaluate the system?

The top row is our protocol. digits.csv is the only data we learn from. We split it once, 80 percent for training and 20 percent for validation. Training fits the weights; validation is what we use to compare variants and pick hyperparameters. Every variant is trained with three different seeds, and we compare the mean.

digits_test.csv plays the role of production. We use it exactly once, on the single model we picked on validation. If we looked at it while tuning, it would stop being an honest estimate of the real world.

For metrics: accuracy is the client's own target, so it is the main number. But accuracy alone can hide a digit the model never gets right, so we also look at recall per class and at the confusion matrix. And we track the training and validation loss every epoch, to see whether the model converges and whether it starts to overfit.

As a preview: the final model scores 96.9 percent on validation and 86.4 percent on the test set, both averaged over three seeds. We will explain that gap at the end.

## Slide 3 · 2B: WHICH VARIANTS DID WE TRY? (Learning rate)

Question 2b: which variants did we try? We changed one thing at a time, starting from the same baseline, and the first one is the learning rate, with plain gradient descent.

On the left is the validation loss per epoch, on a log scale. With 0.01 the curve is still going down after 60 epochs: it learns, but far too slowly. 0.1 is better, and 0.5 and 1 converge fastest and end around 96 percent accuracy, as the bar chart shows.

The interesting case is 3.0. The line is flat at the top: the loss is stuck at 0.5 from the second epoch. What happens is that the first updates are so large that the sigmoid units saturate. Their outputs are pinned at 0 or 1, the derivative of the sigmoid is practically zero there, and so no gradient flows back. The network stops learning completely: the three seeds ended at 0, 12 and 2 percent, which is chance level or worse.

So the learning rate has a window: too small wastes epochs, too large kills the gradient. We kept 0.5 as the baseline for the other studies, and the final model uses 1.0, the other value at the top of that window.

## Slide 4 · 2B: WHICH VARIANTS DID WE TRY? (Architecture)

Next, the architecture. We tried one hidden layer of 16, 32, 64, 128 and 256 units, and two networks with two hidden layers.

Going from 16 to 128 units, the validation accuracy goes up steadily, from 94.5 to 96.6 percent. Adding a second hidden layer did not help: 64-32 and 128-64 are no better than a single layer of 128.

256 units looks bad in the chart, but look at the error bar. Two of the three seeds reached about 96.5 percent; one seed collapsed to 14 percent. That is the same saturation problem as the large learning rate: with more inputs feeding each output unit, the outputs saturate more easily at this learning rate. So a bigger network is not automatically better; it also needs a matching learning rate.

The chart on the right shows the cost. Time per epoch grows with the number of weights, from about 0.03 seconds for 16 units to almost 0.2 for 256. 128 units gave the best accuracy for a reasonable cost, and it became our final architecture.

## Slide 5 · 2B: WHICH VARIANTS DID WE TRY? (Optimization method)

Third, the optimization method. We compared plain gradient descent, momentum and Adam.

They do not share one learning rate: each rule scales its step differently, so we gave each one a rate that suits it. Momentum with alpha 0.9 accumulates roughly ten times the step, so 0.05 for momentum is about the same as 0.5 for gradient descent. Adam normalises the gradient, so its usual value is 0.001.

Gradient descent and Adam follow almost the same loss curve and end at the same accuracy, 96.1 percent.

Momentum behaves differently. In the loss plot you see long flat plateaus followed by sudden drops: the output units are saturated for a while and then escape. In one seed two of the output units never escaped, which means the network never predicts those two digits, and it ends at 74 percent.

So for this problem the optimizer mattered less than the learning rate, and the main risk with momentum here is the extra speed pushing units into saturation.

## Slide 6 · 2B: WHICH VARIANTS DID WE TRY? (Batch size)

Finally, the batch size, which is how we choose between online, mini-batch and batch training. Batch 1 is online learning: one update per sample.

The key to this chart is the number of weight updates per epoch, which is the training set size divided by the batch size: almost 10,000 updates for online, only 20 for batch 512.

The chart on the right shows that online learning is by far the slowest per epoch, about one second, because each update is a tiny matrix product and Python overhead dominates. It is also the noisiest: the error bar is the widest.

Batch 512 is the fastest per epoch, but with only 20 updates per epoch it simply has not converged after 60 epochs, so it ends lowest at 94 percent.

Batch 8 gave the best accuracy, 96.5 percent: many updates per epoch, while still using matrix operations. Batch 32 is close, at 96.1, and more than twice as fast per epoch, but accuracy is the client's target, so the final model uses batch 8.

## Slide 7 · 2: FINAL MODEL IN "PRODUCTION" ([784, 128, 10])

This is our final model. We combined the winner of each study: a learning rate of 1.0 from the top of the learning-rate window, one hidden layer of 128 units, and batch size 8. That combination is better than any single variant from the studies, and it is also the configuration Exercise 3 starts from. We trained it with three seeds and evaluated it once on digits_test.csv, our stand-in for production.

On validation it scores 96.9 percent, plus or minus 0.2 over the three seeds. On the test set it drops to 86.4 percent. The confusion matrix shows where the drop comes from. Every row is close to perfect except one: the row for digit 8 has zero on the diagonal. The model never predicts an 8.

The reason is simple: there is not a single 8 in digits.csv. The network has no output that was ever trained to fire for an 8. 243 of the test images are eights, almost 10 percent, so the best accuracy this model could ever reach on this test set is 90.3 percent, whatever we tune.

If we leave the eights out, the test accuracy is 95.8 percent, close to validation, so the model itself generalises well.

The second weakest digit is 5, with 82 to 86 percent recall depending on the seed, which you can see in the recall chart on the right. As the next slide shows, 5 is also the digit with the fewest training examples.

## Slide 8 · 2: WHY THE TEST SCORE DROPS: THE DATA

So why the gap? It is a data problem, not a model problem.

The chart compares how often each digit appears. In the test set every digit is about 10 percent. In digits.csv, 8 is completely missing and 5 has only 271 images, about 2 percent, against around 1,500 for every other digit.

At the bottom are real test images labelled 8, with what our model predicted. They are clear eights to a human, but the network can only answer with digits it has seen, so it picks the closest shape it knows: mostly 3, then 5, 9 or 2. That matches the confusion matrix on the previous slide.

No learning rate, architecture or optimizer can fix this: you cannot learn a class you have never seen. That is exactly the point of Exercise 3, where the new data in more_digits.csv includes eights and many more fives. Part of the improvement there will come from the data, not only from our techniques.

---

# Probable questions on the code and implementation

Grouped by topic. File references are to `src/perceptrons/`.

## Model and maths

**Q: What are the shapes of the matrices in your forward pass?**
A: Samples are columns. For a batch of n samples, `X` is (784, n). Layer i has `W[i]` of shape (n_out, n_in) and `b[i]` of shape (n_out, 1). Then `h = W @ V + b` is (n_out, n), and numpy broadcasts the bias over the n columns. The output is (10, n), one column per sample (`multilayer.py`, `forward`).

**Q: Write the backpropagation equations you implemented.**
A: Output layer: δ_L = (O − ζ) ⊙ θ'(h_L). Hidden layers: δ_i = (W_{i+1}ᵀ δ_{i+1}) ⊙ θ'(h_i). Gradients: ∂E/∂W_i = δ_i V_{i−1}ᵀ / n and ∂E/∂b_i = (sum of δ_i over the samples) / n. In the code, `backward` returns the deltas and `weight_gradients` turns them into gradients.

**Q: Where does the 1/n come from, and why is it applied in only one place?**
A: The loss is E = 1/(2n) Σ (O − ζ)². `MSE.gradient` returns the per-sample derivative O − ζ without the 1/n, and `weight_gradients` divides by the batch size exactly once. If both divided, the gradient would be n times too small and the effective learning rate would depend on the batch size.

**Q: Your sigmoid is 1/(1 + e^(−2βh)). Why the 2, and what is its derivative?**
A: It follows the course notation: with the 2 it is the logistic version of tanh(βh), rescaled to (0, 1). The derivative is θ'(h) = 2β θ(h)(1 − θ(h)), with a maximum of β/2 at h = 0. That small maximum is one reason our gradients are small and a learning rate of 0.5 works well.

**Q: Why sigmoid outputs with MSE instead of softmax with cross-entropy?**
A: They are what the course covers, and our backprop is checked for that combination. The cost is saturation: with MSE, the output delta contains θ'(h), which is near zero when an output is pinned at 0 or 1. That explains the collapsed runs. Softmax with cross-entropy cancels that factor, so the output delta is just O − ζ. It is a natural next step for Exercise 3.

**Q: How do you turn 10 outputs into a predicted digit?**
A: Argmax over the 10 output units (`one_hot_decode`). Labels are one-hot encoded for training (`one_hot_encode`, which takes `n_classes` explicitly, so a batch that happens to miss a digit still produces 10 rows).

**Q: How did you initialize the weights, and why not with zeros?**
A: Glorot/Xavier uniform: U(−√(6/(n_in + n_out)), +√(6/(n_in + n_out))). With zero weights every hidden unit would compute the same thing and receive the same gradient, so they would never differ (symmetry). The scale keeps h small at the start, so the sigmoids begin in their non-saturated region. Biases start at zero, which is fine because the random weights already break the symmetry.

## Verification

**Q: How do you know your backpropagation is correct?**
A: Three tests in `tests/test_multilayer.py`:
1. A numerical gradient check: every parameter is nudged by ±ε and compared with (E(w + ε) − E(w − ε)) / 2ε.
2. One forward and backward step of a [2, 2, 1] network with fixed weights, compared with numbers we calculated by hand.
3. XOR learned with both [2, 2, 1] and [2, 3, 2, 1].

The optimizers are tested separately on f(w) = (w − 3)², whose minimum we know.

**Q: Why use a central difference in the gradient check?**
A: Its error is O(ε²) instead of O(ε) for a one-sided difference, so it can be compared with a tight tolerance.

## Training loop and optimizers

**Q: How do you switch between online, mini-batch and batch training?**
A: Only through `batch_size`: 1 is online, anything at or above the number of samples is full batch, and anything in between is mini-batch. `iterate_minibatches` shuffles with `rng.permutation` every epoch and keeps the smaller last batch, so every sample is seen once per epoch.

**Q: Explain your momentum and Adam updates.**
A: Momentum: Δw(t) = −η g + α Δw(t−1), then w += Δw. Adam keeps running averages m (of g) and v (of g²), corrects their bias with 1 − β^t, and steps w −= η m̂ / (√v̂ + ε). The state (Δw, m, v, t) lives inside the optimizer object, and `t` counts steps, not parameters (there is a test for this).

**Q: Why does each run create a new optimizer?**
A: Momentum and Adam carry state between steps. Reusing one object would start the next run with the previous run's velocity and moving averages.

**Q: Why did momentum and Adam get different learning rates from gradient descent?**
A: The rules scale the step differently. With α = 0.9, momentum accumulates up to 1/(1 − α) = 10 times the step, so 0.05 for momentum is about 0.5 for gradient descent. Adam divides by √v̂, so its step is roughly η regardless of the gradient size, and 0.001 is its standard value. Giving all three the same η would not be a fair comparison.

**Q: Why do some runs collapse to 14% or 74%?**
A: Saturation. A large update pushes some output units to 0 or 1, where θ' ≈ 0. Their delta is then almost zero, so they stop learning and the network never predicts those digits. Lowering η, using tanh or softmax with cross-entropy, or using a smaller initialization all reduce the risk.

**Q: Is the training reproducible?**
A: Yes. `run_experiment` creates one `np.random.default_rng(seed)` that is used for both initialization and shuffling, and the seed is saved in the config of every result file. The same config gives the same numbers.

**Q: Why no early stopping?**
A: `train` supports a `target_loss` stop, but we used a fixed number of epochs so all variants get the same budget (60 in the studies, 50 for the final model). The final model's validation loss flattens after about epoch 30 and does not rise, so there was no overfitting to stop.

## Evaluation and data handling

**Q: How is the validation set built? Is it stratified?**
A: The first 80% of `digits.csv` is training and the last 20% is validation (`split_train_and_validation`). The file is already shuffled, and the class proportions in the two parts are similar, but the split is not explicitly stratified. With more time we would shuffle with the seed, stratify, or use k-fold cross-validation.

**Q: Why use the test set only once?**
A: If we chose hyperparameters by looking at `digits_test.csv`, its score would be optimistically biased and would no longer estimate real-world performance. The assignment explicitly treats it as production.

**Q: How did you choose the final model?**
A: Each study changes one hyperparameter from a common baseline and picks a winner by mean validation accuracy over 3 seeds: lr 0.5–1, 128 hidden units, batch size 8. The final model combines those winners (`FINAL` in `ex2.py`) and is confirmed on validation over 3 seeds: 96.85 ± 0.21%, against 96.60% for the best single variant. Averaging over seeds stops one lucky initialization from winning. It is also the configuration Exercise 3 starts from.

**Q: The learning-rate study used 0.5 as the baseline. Why does the final model use 1.0?**
A: In the study, 0.5 and 1 were tied at the top (96.1% and 96.2%), so either was a reasonable choice. The final configuration is the one the group agreed on and Exercise 3 builds on, where lr 1.0 at batch 8 was compared with lower rates. We did not re-run that comparison in our own studies, so we confirm the combined configuration over 3 seeds instead. 3.0 is where training breaks.

**Q: A one-factor-at-a-time search assumes the hyperparameters are independent. Are they?**
A: No. The best learning rate depends on the optimizer, the batch size and the activation. That is why each optimizer got its own learning rate, and why the combined configuration is checked again over 3 seeds rather than assumed to be best. A full grid search would be more thorough but would cost the product of all the grid sizes in runs.

**Q: Why accuracy, and not F1 or something else?**
A: It is the client's target, and the classes in the test set are balanced (about 10% each), so accuracy is not misleading on its own. We add per-class recall and the confusion matrix because accuracy alone can hide a class that is always wrong, which is exactly what happened with the 8s.

**Q: Why can the network never predict an 8?**
A: Output unit 8 receives target 0 for every training sample, so training pushes its weights and bias towards always outputting about 0. It is never the largest output, so argmax never picks it.

**Q: Did you normalize the inputs?**
A: The pixels are already in [0, 1] in the CSV, which we checked when loading, so no extra scaling was needed. Large unscaled inputs (0–255) would saturate the first layer immediately.

**Q: What does one epoch cost?**
A: Roughly the number of samples times the number of weights, O(n · Σ n_l n_{l+1}), for each of the forward and backward passes. The measured time per epoch grows with the layer sizes accordingly: about 0.03 s for 16 hidden units, almost 0.2 s for 256.

**Q: Why is the confusion matrix row-normalized?**
A: Each row then sums to 1, so the diagonal is the recall of that class, and digits with different sample counts are comparable.

**Q: How would you reach 98%?**
A: Mainly more and better data, especially examples of 8 and 5. That is Exercise 3. On the model side: softmax with cross-entropy, a tuned learning rate or a learning-rate schedule, and more epochs.
