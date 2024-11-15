# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

## Parallel Analytics for Task 3.1 and 3.2
```
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
C:\Users\ASUS\Documents\Project\mini-torch\mod3-ethan-yz-
hao\minitorch\fast_ops.py (163)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, C:\Users\ASUS\Documents\Project\mini-torch\mod3-ethan-yz-hao\minitorch\fast_ops.py (163)
-------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                              |
        out: Storage,                                                                      |
        out_shape: Shape,                                                                  |
        out_strides: Strides,                                                              |
        in_storage: Storage,                                                               |
        in_shape: Shape,                                                                   |
        in_strides: Strides,                                                               |
    ) -> None:                                                                             |
        # TODO: Implement for Task 3.1.                                                    |
        if list(in_shape) == list(out_shape) and list(in_strides) == list(out_strides):    |
            for i in prange(len(out)):-----------------------------------------------------| #0
                out[i] = fn(in_storage[i])                                                 |
        else:                                                                              |
            for i in prange(len(out)):-----------------------------------------------------| #1
                in_i = np.empty(MAX_DIMS, np.int32)                                        |
                out_i = np.empty(MAX_DIMS, np.int32)                                       |
                                                                                           |
                to_index(i, out_shape, out_i)                                              |
                broadcast_index(out_i, out_shape, in_shape, in_i)                          |
                                                                                           |
                out_pos = index_to_position(out_i, out_strides)                            |
                in_pos = index_to_position(in_i, in_strides)                               |
                                                                                           |
                out[out_pos] = fn(in_storage[in_pos])                                      |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #0, #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
C:\Users\ASUS\Documents\Project\mini-torch\mod3-ethan-yz-
hao\minitorch\fast_ops.py (177) is hoisted out of the parallel loop labelled #1
(it will be performed before the loop is executed and reused inside the loop):
   Allocation:: in_i = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
C:\Users\ASUS\Documents\Project\mini-torch\mod3-ethan-yz-
hao\minitorch\fast_ops.py (178) is hoisted out of the parallel loop labelled #1
(it will be performed before the loop is executed and reused inside the loop):
   Allocation:: out_i = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
C:\Users\ASUS\Documents\Project\mini-torch\mod3-ethan-yz-
hao\minitorch\fast_ops.py (214)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, C:\Users\ASUS\Documents\Project\mini-torch\mod3-ethan-yz-hao\minitorch\fast_ops.py (214)
-----------------------------------------------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                                                                |
        out: Storage,                                                                                                        |
        out_shape: Shape,                                                                                                    |
        out_strides: Strides,                                                                                                |
        a_storage: Storage,                                                                                                  |
        a_shape: Shape,                                                                                                      |
        a_strides: Strides,                                                                                                  |
        b_storage: Storage,                                                                                                  |
        b_shape: Shape,                                                                                                      |
        b_strides: Strides,                                                                                                  |
    ) -> None:                                                                                                               |
        # TODO: Implement for Task 3.1.                                                                                      |
        if list(a_strides) == list(b_strides) == list(out_strides) and list(a_shape) == list(b_shape) == list(out_shape):    |
            for i in prange(len(out)):---------------------------------------------------------------------------------------| #2
                out[i] = fn(a_storage[i], b_storage[i])                                                                      |
                                                                                                                             |
        else:                                                                                                                |
            for i in prange(len(out)):---------------------------------------------------------------------------------------| #3
                out_i: Index = np.empty(MAX_DIMS, np.int32)                                                                  |
                a_i: Index = np.empty(MAX_DIMS, np.int32)                                                                    |
                b_i: Index = np.empty(MAX_DIMS, np.int32)                                                                    |
                                                                                                                             |
                to_index(i, out_shape, out_i)                                                                                |
                broadcast_index(out_i, out_shape, a_shape, a_i)                                                              |
                broadcast_index(out_i, out_shape, b_shape, b_i)                                                              |
                                                                                                                             |
                out_pos = index_to_position(out_i, out_strides)                                                              |
                a_pos = index_to_position(a_i, a_strides)                                                                    |
                b_pos = index_to_position(b_i, b_strides)                                                                    |
                                                                                                                             |
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])                                                        |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
C:\Users\ASUS\Documents\Project\mini-torch\mod3-ethan-yz-
hao\minitorch\fast_ops.py (232) is hoisted out of the parallel loop labelled #3
(it will be performed before the loop is executed and reused inside the loop):
   Allocation:: out_i: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
C:\Users\ASUS\Documents\Project\mini-torch\mod3-ethan-yz-
hao\minitorch\fast_ops.py (233) is hoisted out of the parallel loop labelled #3
(it will be performed before the loop is executed and reused inside the loop):
   Allocation:: a_i: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
C:\Users\ASUS\Documents\Project\mini-torch\mod3-ethan-yz-
hao\minitorch\fast_ops.py (234) is hoisted out of the parallel loop labelled #3
(it will be performed before the loop is executed and reused inside the loop):
   Allocation:: b_i: Index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
C:\Users\ASUS\Documents\Project\mini-torch\mod3-ethan-yz-
hao\minitorch\fast_ops.py (270)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, C:\Users\ASUS\Documents\Project\mini-torch\mod3-ethan-yz-hao\minitorch\fast_ops.py (270)
---------------------------------------------------------------|loop #ID
    def _reduce(                                               |
        out: Storage,                                          |
        out_shape: Shape,                                      |
        out_strides: Strides,                                  |
        a_storage: Storage,                                    |
        a_shape: Shape,                                        |
        a_strides: Strides,                                    |
        reduce_dim: int,                                       |
    ) -> None:                                                 |
        # TODO: Implement for Task 3.1.                        |
        reduce_size = a_shape[reduce_dim]                      |
        reduce_stride = a_strides[reduce_dim]                  |
        for i in prange(len(out)):-----------------------------| #4
            out_i = np.empty(MAX_DIMS, np.int32)               |
                                                               |
            to_index(i, out_shape, out_i)                      |
                                                               |
            out_pos = index_to_position(out_i, out_strides)    |
            a_pos = index_to_position(out_i, a_strides)        |
                                                               |
            temp = out[out_pos]                                |
            for _ in range(reduce_size):                       |
                temp = fn(temp, a_storage[a_pos])              |
                a_pos += reduce_stride                         |
            out[out_pos] = temp                                |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #4).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
C:\Users\ASUS\Documents\Project\mini-torch\mod3-ethan-yz-
hao\minitorch\fast_ops.py (283) is hoisted out of the parallel loop labelled #4
(it will be performed before the loop is executed and reused inside the loop):
   Allocation:: out_i = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
C:\Users\ASUS\Documents\Project\mini-torch\mod3-ethan-yz-
hao\minitorch\fast_ops.py (299)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, C:\Users\ASUS\Documents\Project\mini-torch\mod3-ethan-yz-hao\minitorch\fast_ops.py (299)
----------------------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                              |
    out: Storage,                                                                                         |
    out_shape: Shape,                                                                                     |
    out_strides: Strides,                                                                                 |
    a_storage: Storage,                                                                                   |
    a_shape: Shape,                                                                                       |
    a_strides: Strides,                                                                                   |
    b_storage: Storage,                                                                                   |
    b_shape: Shape,                                                                                       |
    b_strides: Strides,                                                                                   |
) -> None:                                                                                                |
    """NUMBA tensor matrix multiply function.                                                             |
                                                                                                          |
    Should work for any tensor shapes that broadcast as long as                                           |
                                                                                                          |
    ```                                                                                                   |
    assert a_shape[-1] == b_shape[-2]                                                                     |
    ```                                                                                                   |
                                                                                                          |
    Optimizations:                                                                                        |
                                                                                                          |
    * Outer loop in parallel                                                                              |
    * No index buffers or function calls                                                                  |
    * Inner loop should have no global writes, 1 multiply.                                                |
                                                                                                          |
                                                                                                          |
    Args:                                                                                                 |
    ----                                                                                                  |
        out (Storage): storage for `out` tensor                                                           |
        out_shape (Shape): shape for `out` tensor                                                         |
        out_strides (Strides): strides for `out` tensor                                                   |
        a_storage (Storage): storage for `a` tensor                                                       |
        a_shape (Shape): shape for `a` tensor                                                             |
        a_strides (Strides): strides for `a` tensor                                                       |
        b_storage (Storage): storage for `b` tensor                                                       |
        b_shape (Shape): shape for `b` tensor                                                             |
        b_strides (Strides): strides for `b` tensor                                                       |
                                                                                                          |
    Returns:                                                                                              |
    -------                                                                                               |
        None : Fills in `out`                                                                             |
                                                                                                          |
    """                                                                                                   |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                                |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                                |
                                                                                                          |
    # TODO: Implement for Task 3.2.                                                                       |
    row_a = a_strides[2]                                                                                  |
    col_b = b_strides[1]                                                                                  |
    blocks = a_shape[-1]                                                                                  |
                                                                                                          |
    for row_i in prange(0, out_shape[0]):-----------------------------------------------------------------| #5
        for col_j in range(0, out_shape[1]):                                                              |
            for block_k in range(0, out_shape[2]):                                                        |
                row_s = row_i * a_batch_stride + col_j * a_strides[1]                                     |
                col_s = row_i * b_batch_stride + block_k * b_strides[2]                                   |
                                                                                                          |
                temp = 0.0                                                                                |
                for _ in range(0, blocks):                                                                |
                    temp += a_storage[row_s] * b_storage[col_s]                                           |
                    row_s += row_a                                                                        |
                    col_s += col_b                                                                        |
                                                                                                          |
                out[row_i * out_strides[0] + col_j * out_strides[1] + block_k * out_strides[2]] = temp    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #5).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```