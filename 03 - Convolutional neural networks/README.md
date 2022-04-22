# Convolutional neural networks

## Assignment
Ta utgangspunkt i nn.py eller nn_sequential.py i [ntnu-tdat3025/cnn/mnist](https://gitlab.com/ntnu-tdat3025/cnn/mnist). Kjør først dette eksempelet og se hva accuracy modellen oppnår.
a) Utvid modellen som vist nedenfor. Ca hva accuracy oppnår denne modellen?

## Results
### Example/Benchmark
The example from "examples/nn_sequential" uses the following model:
```python
self.logits = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5, padding=2),
                            nn.MaxPool2d(kernel_size=2),
                            nn.Flatten(),
                            nn.Linear(32 * 14 * 14, 10))
```
Running the code gives the following output:

|*Epoch*| *Accuracy*|
|:---:|:---:|
|1|94.65%|
|2|96.90%|
|3|97.73%|
|4|97.99%|
|5|98.13%|

### Part A
Improving the model from "examples/nn_sequential" by expanding the model to
```python
self.logits = nn.Sequential(
              nn.Conv2d(1, 32, kernel_size=5, padding=2),
              nn.MaxPool2d(kernel_size=2),
                # Adding this part
                nn.Conv2d(32, 64, kernel_size=5, padding=2),
                nn.MaxPool2d(kernel_size=2),
              nn.Flatten(),
                # Changing this part
                nn.Linear(64 * 7 * 7, 10))
```

which gives us the following result:

|*Epoch*|*Accuracy*|
|:---:|:---:|
|1|97.02%|
|2|97.98%|
|3|98.34%|
|4|98.52%|
|5|98.57%|

As we can see the best accuracy has gone up from 98.13% to 98.57%

### Part B
Continuiong with the model from Part A we expand it again
```python
self.logits = nn.Sequential(
              nn.Conv2d(1, 32, kernel_size=5, padding=2),
              nn.MaxPool2d(kernel_size=2),
              nn.Conv2d(32, 64, kernel_size=5, padding=2),
              nn.MaxPool2d(kernel_size=2),
              nn.Flatten(),
              nn.Linear(64 * 7 * 7, 1024),
                # Adding this part
                nn.Linear(1024 * 1 * 1, 10))
```
Which gives the following output:

|*Epoch*|*Accuracy*|
|:---:|:---:|
|1|97.66%|
|2|98.18%|
|3|98.38%|
|4|98.27%|
|5|98.07%|

As we can see, this expansion seems to make little difference on the accuracy compared to part A.

### Part C
To further enhance the model, i use Part B as a base, then use ReLU and Dropout in an attempt to improve the accuracy of the model.
ReLU enhances the input making features stand out more, while dropout randomly removes nodes forcing to model to be more genralized (prevents the model from remembering the exact training data).

My model ended up looking like this:

```python
self.logits = nn.Sequential(
              nn.Conv2d(1, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Dropout(p=0.3),
              nn.MaxPool2d(kernel_size=2),
              nn.Conv2d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(),
              nn.MaxPool2d(kernel_size=2),
              nn.Flatten(),
              nn.Linear(64 * 7 * 7, 1024),
              nn.Linear(1024 * 1 * 1, 10))
```
And gave the following result:

|*Epoch*|*Accuracy*|
|:---:|:---:|
|1|97.57%|
|2|98.11%|
|3|98.66%|
|4|98.80%|
|5|98.62%|

which is marginally better than Part A and B, but a lot better than the Benchmark.

### Part D
Here i simply switch out the MNIST data set (of numbers), with FashionMNIST which is of different clothing articles. These should be much more difficult but lets see how high accuracy i can get.

Using this model:
```python
self.logits = nn.Sequential(
              nn.Conv2d(1, 32, kernel_size=5, padding=2),
              nn.MaxPool2d(kernel_size=2),
              nn.ReLU(),
              nn.Dropout(p=0.2),
              nn.Conv2d(32, 64, kernel_size=5, padding=2),
              nn.ReLU(),
              nn.MaxPool2d(kernel_size=2),
              nn.Flatten(),
              nn.Linear(64 * 7 * 7, 10))
```
I get the following results

|*Epoch*|*Accuracy*|
|:---:|:---:|
|1|83.52%|
|2|86.68%|
|3|87.68%|
|4|88.53%|
|5|89.30%|

<!--
And for fun i decided to visualize how the model changes from epoch to epoch with this gif:
[visualization gif]()
-->
