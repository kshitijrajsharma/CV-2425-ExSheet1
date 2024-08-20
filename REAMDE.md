# Exercise set 1

All exercises need to be implemented within the function skeletons found in `tensors.py`.
To test the functionality of your code, you can use 
`pytest` (which is also used for autograding). Install `pytest` (in your conda environment) via

```bash
pip install pytest
```

You can then run 

```bash
pytest tensors_test.py
```
which will run all 3 tests. If you only want to test, e.g., Exercise 1, call `pytest` as follows:

```bash
pytest tensors_test.py::test_ex1
```
---

### Exercise 1

Create a `torch` 32-bit floating point tensor that holds a sequence of integers from `0` to `16*3*32*32`, convert the tensor to 32-bit floating point (`torch.float32`) and reshape the tensor to shape  `(16,3,32,32)`. Implement these steps in the function `assignment_ex1` and return the tensor.

### Exercise 2

Create a `torch` 32-bit floating point tensor `T0`   that holds a sequence of integers from `0` to `16*3*3` and reshape that tensor to shape `(16,3,3)`. Then, create a second tensor `T1` which is `T0` multiplied by 3. Finally, use `torch.matmul` to multiply all 16 3x3 matrices from `T0` with the 16 3x3 matrices in `T1` and return the result. Implement the functionality within `assignment_ex2`.

### Exericse 3

Create a `torch` 32-bit floating point tensor `T0`   that holds a sequence of integers from `0` to `16*3*3` and reshape that tensor to shape `(16,3,3)`. Return the tensor where we switch all rows and columns of all 3x3 matrices. Implement this functionality in `assignment_ex3` and use `einops.rearrange`.

**Note**: To test this locally on your system, e.g., in your conda environment (which already holds your PyTorch installation), install `einops` via

```bash
pip install einops
```


