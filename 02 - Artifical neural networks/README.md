# Assignment 2 - Artifical neural networks

### NOT - operator

![](./images/a_plot.png)

### NAND - operator
![](./images/b_plot.png)

### XOR - operator
**Decent starting values**

```python
W1_init = [[1.0, -1.0], [1.0, -1.0]]
W2_init = [[-1.0], [-1.0]]
b1_init = [[1.0, 1.0]]
b2_init = [[-1.0]]
```

![](./images/c_plot.png)

**Bad starting values**

```python
W1_init = [[1.0, -1.0], [1.0, 1.0]]
W2_init = [[-1.0], [-1.0]]
b1_init = [[-1.0, -1.0]]
b2_init = [[-1.0]]
```



![](./images/c_plot_bad.png)

### Number recognision

The following shows a gif visualizing how W changes depending on how many epochs the neural network runs:

![oppgD gif](./images/d_visualization.gif)


