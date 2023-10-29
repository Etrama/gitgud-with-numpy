# First 50 numpy exercises

This is a set of exercises collected by [Rougier](https://github.com/rougier/numpy-100) in the numpy maling list, on stack overflow and in the numpy documentation. 

All credits to Rougier for curating this list. I am simply trying to solve it for practice and hoping it serves as a reference for others. I am surprised I didn't come across it before.

This is intended to serve as a stepping stone to becoming a better Data Scientist / Machine Learning Researcher. I came across this list on Neel Nanda's page titled ["A Barebones Guide to Mechanistic Interpretability Prerequisites"](https://www.neelnanda.io/mechanistic-interpretability/prereqs) in the hopes of being a better researcher and maybe someday working on mechanistic interpretability.

#### 1. Import the numpy package under the name `np` (★☆☆)


```python
import numpy as np
```

#### 2. Print the numpy version and the configuration (★☆☆)


```python
print(f"{np.__version__=}")
print(np.show_config())
```

    np.__version__='1.26.0'
    Build Dependencies:
      blas:
        detection method: pkgconfig
        found: true
        include directory: C:/Users/kaush/anaconda3/envs/stats/Library/include
        lib directory: C:/Users/kaush/anaconda3/envs/stats/Library/lib
        name: mkl-sdl
        pc file directory: C:\b\abs_9fu2cs2527\croot\numpy_and_numpy_base_1695830496596\_h_env\Library\lib\pkgconfig
        version: '2023.1'
      lapack:
        detection method: pkgconfig
        found: true
        include directory: C:/Users/kaush/anaconda3/envs/stats/Library/include
        lib directory: C:/Users/kaush/anaconda3/envs/stats/Library/lib
        name: mkl-sdl
        pc file directory: C:\b\abs_9fu2cs2527\croot\numpy_and_numpy_base_1695830496596\_h_env\Library\lib\pkgconfig
        version: '2023.1'
    Compilers:
      c:
        commands: cl.exe
        linker: link
        name: msvc
        version: 19.29.30152
      c++:
        commands: cl.exe
        linker: link
        name: msvc
        version: 19.29.30152
      cython:
        commands: cython
        linker: cython
        name: cython
        version: 3.0.0
    Machine Information:
      build:
        cpu: x86_64
        endian: little
        family: x86_64
        system: windows
      host:
        cpu: x86_64
        endian: little
        family: x86_64
        system: windows
    Python Information:
      path: C:\b\abs_9fu2cs2527\croot\numpy_and_numpy_base_1695830496596\_h_env\python.exe
      version: '3.11'
    SIMD Extensions:
      baseline:
      - SSE
      - SSE2
      - SSE3
      found:
      - SSSE3
      - SSE41
      - POPCNT
      - SSE42
      - AVX
      - F16C
      - FMA3
      - AVX2
      not found:
      - AVX512F
      - AVX512CD
      - AVX512_SKX
      - AVX512_CLX
      - AVX512_CNL
      - AVX512_ICL
    
    None
    

#### 3. Create a null vector of size 10 (★☆☆)


```python
np.zeros(10)
```




    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])




```python
null_vector = np.zeros(10)
null_vector

```




    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])



I prefer the non-print version of the output  
the print version loses the commas and the array(*) notation

#### 4. How to find the memory size of any array (★☆☆)


```python
null_vector = np.zeros(10)
print(f"{null_vector.size = }")
print(f"{null_vector.itemsize = }")
print(f"Total size of null_vector = {null_vector.size * null_vector.itemsize} bytes")
```

    null_vector.size = 10
    null_vector.itemsize = 8
    Total size of null_vector = 80 bytes
    

#### 5. How to get the documentation of the numpy add function from the command line? (★☆☆)

There are 3 ways to do this, the first two ways with the "?" only apply to notebooks I believe, but I could be wrong. This might be the best thing you learn from this notebook if you weren't aware of it before.


```python
np.add?
```

    [1;31mCall signature:[0m  [0mnp[0m[1;33m.[0m[0madd[0m[1;33m([0m[1;33m*[0m[0margs[0m[1;33m,[0m [1;33m**[0m[0mkwargs[0m[1;33m)[0m[1;33m[0m[1;33m[0m[0m
    [1;31mType:[0m            ufunc
    [1;31mString form:[0m     <ufunc 'add'>
    [1;31mFile:[0m            c:\users\kaush\anaconda3\envs\stats\lib\site-packages\numpy\__init__.py
    [1;31mDocstring:[0m      
    add(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])
    
    Add arguments element-wise.
    
    Parameters
    ----------
    x1, x2 : array_like
        The arrays to be added.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.
    
    Returns
    -------
    add : ndarray or scalar
        The sum of `x1` and `x2`, element-wise.
        This is a scalar if both `x1` and `x2` are scalars.
    
    Notes
    -----
    Equivalent to `x1` + `x2` in terms of array broadcasting.
    
    Examples
    --------
    >>> np.add(1.0, 4.0)
    5.0
    >>> x1 = np.arange(9.0).reshape((3, 3))
    >>> x2 = np.arange(3.0)
    >>> np.add(x1, x2)
    array([[  0.,   2.,   4.],
           [  3.,   5.,   7.],
           [  6.,   8.,  10.]])
    
    The ``+`` operator can be used as a shorthand for ``np.add`` on ndarrays.
    
    >>> x1 = np.arange(9.0).reshape((3, 3))
    >>> x2 = np.arange(3.0)
    >>> x1 + x2
    array([[ 0.,  2.,  4.],
           [ 3.,  5.,  7.],
           [ 6.,  8., 10.]])
    [1;31mClass docstring:[0m
    Functions that operate element by element on whole arrays.
    
    To see the documentation for a specific ufunc, use `info`.  For
    example, ``np.info(np.sin)``.  Because ufuncs are written in C
    (for speed) and linked into Python with NumPy's ufunc facility,
    Python's help() function finds this page whenever help() is called
    on a ufunc.
    
    A detailed explanation of ufuncs can be found in the docs for :ref:`ufuncs`.
    
    **Calling ufuncs:** ``op(*x[, out], where=True, **kwargs)``
    
    Apply `op` to the arguments `*x` elementwise, broadcasting the arguments.
    
    The broadcasting rules are:
    
    * Dimensions of length 1 may be prepended to either array.
    * Arrays may be repeated along dimensions of length 1.
    
    Parameters
    ----------
    *x : array_like
        Input arrays.
    out : ndarray, None, or tuple of ndarray and None, optional
        Alternate array object(s) in which to put the result; if provided, it
        must have a shape that the inputs broadcast to. A tuple of arrays
        (possible only as a keyword argument) must have length equal to the
        number of outputs; use None for uninitialized outputs to be
        allocated by the ufunc.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the :ref:`ufunc docs <ufuncs.kwargs>`.
    
    Returns
    -------
    r : ndarray or tuple of ndarray
        `r` will have the shape that the arrays in `x` broadcast to; if `out` is
        provided, it will be returned. If not, `r` will be allocated and
        may contain uninitialized values. If the function has more than one
        output, then the result will be a tuple of arrays.


```python
np.add??
```

    [1;31mCall signature:[0m  [0mnp[0m[1;33m.[0m[0madd[0m[1;33m([0m[1;33m*[0m[0margs[0m[1;33m,[0m [1;33m**[0m[0mkwargs[0m[1;33m)[0m[1;33m[0m[1;33m[0m[0m
    [1;31mType:[0m            ufunc
    [1;31mString form:[0m     <ufunc 'add'>
    [1;31mFile:[0m            c:\users\kaush\anaconda3\envs\stats\lib\site-packages\numpy\__init__.py
    [1;31mDocstring:[0m      
    add(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])
    
    Add arguments element-wise.
    
    Parameters
    ----------
    x1, x2 : array_like
        The arrays to be added.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.
    
    Returns
    -------
    add : ndarray or scalar
        The sum of `x1` and `x2`, element-wise.
        This is a scalar if both `x1` and `x2` are scalars.
    
    Notes
    -----
    Equivalent to `x1` + `x2` in terms of array broadcasting.
    
    Examples
    --------
    >>> np.add(1.0, 4.0)
    5.0
    >>> x1 = np.arange(9.0).reshape((3, 3))
    >>> x2 = np.arange(3.0)
    >>> np.add(x1, x2)
    array([[  0.,   2.,   4.],
           [  3.,   5.,   7.],
           [  6.,   8.,  10.]])
    
    The ``+`` operator can be used as a shorthand for ``np.add`` on ndarrays.
    
    >>> x1 = np.arange(9.0).reshape((3, 3))
    >>> x2 = np.arange(3.0)
    >>> x1 + x2
    array([[ 0.,  2.,  4.],
           [ 3.,  5.,  7.],
           [ 6.,  8., 10.]])
    [1;31mClass docstring:[0m
    Functions that operate element by element on whole arrays.
    
    To see the documentation for a specific ufunc, use `info`.  For
    example, ``np.info(np.sin)``.  Because ufuncs are written in C
    (for speed) and linked into Python with NumPy's ufunc facility,
    Python's help() function finds this page whenever help() is called
    on a ufunc.
    
    A detailed explanation of ufuncs can be found in the docs for :ref:`ufuncs`.
    
    **Calling ufuncs:** ``op(*x[, out], where=True, **kwargs)``
    
    Apply `op` to the arguments `*x` elementwise, broadcasting the arguments.
    
    The broadcasting rules are:
    
    * Dimensions of length 1 may be prepended to either array.
    * Arrays may be repeated along dimensions of length 1.
    
    Parameters
    ----------
    *x : array_like
        Input arrays.
    out : ndarray, None, or tuple of ndarray and None, optional
        Alternate array object(s) in which to put the result; if provided, it
        must have a shape that the inputs broadcast to. A tuple of arrays
        (possible only as a keyword argument) must have length equal to the
        number of outputs; use None for uninitialized outputs to be
        allocated by the ufunc.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the :ref:`ufunc docs <ufuncs.kwargs>`.
    
    Returns
    -------
    r : ndarray or tuple of ndarray
        `r` will have the shape that the arrays in `x` broadcast to; if `out` is
        provided, it will be returned. If not, `r` will be allocated and
        may contain uninitialized values. If the function has more than one
        output, then the result will be a tuple of arrays.


```python
np.info(np.add)
```

    add(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj])
    
    Add arguments element-wise.
    
    Parameters
    ----------
    x1, x2 : array_like
        The arrays to be added.
        If ``x1.shape != x2.shape``, they must be broadcastable to a common
        shape (which becomes the shape of the output).
    out : ndarray, None, or tuple of ndarray and None, optional
        A location into which the result is stored. If provided, it must have
        a shape that the inputs broadcast to. If not provided or None,
        a freshly-allocated array is returned. A tuple (possible only as a
        keyword argument) must have length equal to the number of outputs.
    where : array_like, optional
        This condition is broadcast over the input. At locations where the
        condition is True, the `out` array will be set to the ufunc result.
        Elsewhere, the `out` array will retain its original value.
        Note that if an uninitialized `out` array is created via the default
        ``out=None``, locations within it where the condition is False will
        remain uninitialized.
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.
    
    Returns
    -------
    add : ndarray or scalar
        The sum of `x1` and `x2`, element-wise.
        This is a scalar if both `x1` and `x2` are scalars.
    
    Notes
    -----
    Equivalent to `x1` + `x2` in terms of array broadcasting.
    
    Examples
    --------
    >>> np.add(1.0, 4.0)
    5.0
    >>> x1 = np.arange(9.0).reshape((3, 3))
    >>> x2 = np.arange(3.0)
    >>> np.add(x1, x2)
    array([[  0.,   2.,   4.],
           [  3.,   5.,   7.],
           [  6.,   8.,  10.]])
    
    The ``+`` operator can be used as a shorthand for ``np.add`` on ndarrays.
    
    >>> x1 = np.arange(9.0).reshape((3, 3))
    >>> x2 = np.arange(3.0)
    >>> x1 + x2
    array([[ 0.,  2.,  4.],
           [ 3.,  5.,  7.],
           [ 6.,  8., 10.]])
    

#### 6. Create a null vector of size 10 but the fifth value which is 1 (★☆☆)


```python
null_vector = np.zeros(10)
null_vector[4] = 1
null_vector
```




    array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])



#### 7. Create a vector with values ranging from 10 to 49 (★☆☆)

I'm not sure if the vector should contain random values from 10 to 40 or sequential values from 10 to 49. So we'll do both.

#### Sequential Value Version


```python
seq_vec = np.arange(10,50)
seq_vec
```




    array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
           27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
           44, 45, 46, 47, 48, 49])



#### Random Integer Version


```python
rand_vec = np.random.randint(low=10,high=50,size=20)
# remember that the low is inclusive and the high is exclusive
rand_vec
```




    array([37, 47, 30, 44, 40, 47, 18, 10, 20, 41, 26, 39, 21, 39, 29, 20, 46,
           36, 36, 20])



#### 8. Reverse a vector (first element becomes last) (★☆☆)


```python
seq_vec = np.arange(0,9)
seq_vec[::-1]
```




    array([8, 7, 6, 5, 4, 3, 2, 1, 0])



##### Let's see what happens with a matrix:


```python
seq_vec = np.arange(0,9)
seq_vec = seq_vec.reshape(3, 3)
seq_vec
```




    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])




```python
seq_vec[::-1]
```




    array([[6, 7, 8],
           [3, 4, 5],
           [0, 1, 2]])



#### Finally, let's see what happens with a tensor for good measure:


```python
tensor = np.random.randint(low=0,high=10,size=50)
tensor
```




    array([4, 8, 3, 5, 4, 5, 3, 2, 1, 1, 4, 2, 9, 5, 7, 8, 4, 2, 3, 3, 2, 5,
           8, 9, 3, 1, 6, 1, 8, 9, 7, 2, 6, 7, 6, 3, 2, 1, 1, 4, 9, 6, 8, 6,
           8, 0, 9, 3, 3, 3])




```python
tensor = tensor.reshape(5, 5, 2)
tensor
```




    array([[[4, 8],
            [3, 5],
            [4, 5],
            [3, 2],
            [1, 1]],
    
           [[4, 2],
            [9, 5],
            [7, 8],
            [4, 2],
            [3, 3]],
    
           [[2, 5],
            [8, 9],
            [3, 1],
            [6, 1],
            [8, 9]],
    
           [[7, 2],
            [6, 7],
            [6, 3],
            [2, 1],
            [1, 4]],
    
           [[9, 6],
            [8, 6],
            [8, 0],
            [9, 3],
            [3, 3]]])




```python
tensor[::-1]
```




    array([[[9, 6],
            [8, 6],
            [8, 0],
            [9, 3],
            [3, 3]],
    
           [[7, 2],
            [6, 7],
            [6, 3],
            [2, 1],
            [1, 4]],
    
           [[2, 5],
            [8, 9],
            [3, 1],
            [6, 1],
            [8, 9]],
    
           [[4, 2],
            [9, 5],
            [7, 8],
            [4, 2],
            [3, 3]],
    
           [[4, 8],
            [3, 5],
            [4, 5],
            [3, 2],
            [1, 1]]])



So we see that for a vector we reverse the elements, for a matrx we reverse the row order and for a tensor we reverse the matrix order.

#### 9. Create a 3x3 matrix with values ranging from 0 to 8 (★☆☆)


```python
matrix = np.arange(0,9).reshape(3,3)
matrix
```




    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])



#### 10. Find indices of non-zero elements from [1,2,0,0,4,0] (★☆☆)


```python
vector = np.array([1,2,0,0,4,0])
vector.nonzero()
```




    (array([0, 1, 4], dtype=int64),)




```python
vector[vector.nonzero()]
```




    array([1, 2, 4])



#### 11. Create a 3x3 identity matrix (★☆☆)


```python
identity_matrix = np.eye(3)
identity_matrix
```




    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])



#### 12. Create a 3x3x3 array with random values (★☆☆)


```python
random_int_tensor = np.random.randint(low=0,high=10,size=(3,3,3))
random_int_tensor
```




    array([[[1, 7, 6],
            [0, 5, 8],
            [1, 1, 4]],
    
           [[9, 4, 6],
            [3, 0, 0],
            [7, 6, 9]],
    
           [[9, 6, 4],
            [4, 9, 2],
            [0, 5, 1]]])




```python
random_tensor = np.random.random((3,3,3))
random_tensor
```




    array([[[0.10881079, 0.82777305, 0.73024425],
            [0.41001143, 0.6864714 , 0.8286755 ],
            [0.45317059, 0.731864  , 0.24504508]],
    
           [[0.27105589, 0.31979774, 0.53269988],
            [0.74576895, 0.50118312, 0.91641389],
            [0.47912516, 0.88949762, 0.97725423]],
    
           [[0.18685545, 0.84190864, 0.59051282],
            [0.50079992, 0.10817144, 0.95391992],
            [0.03638185, 0.52208639, 0.48453625]]])



#### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)


```python
random_array = np.random.random((10,10))
random_array
```




    array([[0.54719157, 0.83132803, 0.43281279, 0.19023279, 0.53131627,
            0.72643636, 0.68634597, 0.61568373, 0.04141836, 0.87535182],
           [0.56288263, 0.89685912, 0.14898017, 0.14752818, 0.40686519,
            0.22272535, 0.97574528, 0.37510763, 0.22641714, 0.69185797],
           [0.42784303, 0.41151925, 0.32606156, 0.0714769 , 0.77423661,
            0.74242478, 0.5406395 , 0.221898  , 0.50075341, 0.2534813 ],
           [0.34410587, 0.24116138, 0.18034516, 0.09613779, 0.42510876,
            0.65994404, 0.80072605, 0.09166011, 0.61156817, 0.92289304],
           [0.24209721, 0.91770228, 0.19088493, 0.60953766, 0.57709861,
            0.85253104, 0.06680409, 0.30503953, 0.2193305 , 0.77070331],
           [0.75137254, 0.02115599, 0.91557485, 0.80031674, 0.29321583,
            0.16392821, 0.27879865, 0.06015219, 0.20942407, 0.13387668],
           [0.48156959, 0.55267476, 0.66147292, 0.49684291, 0.23053481,
            0.42214255, 0.46477693, 0.84837187, 0.17167265, 0.53384572],
           [0.16045123, 0.37403666, 0.82160964, 0.36511628, 0.84468648,
            0.54006019, 0.86844253, 0.90649856, 0.94265222, 0.33791684],
           [0.77838936, 0.66922879, 0.81474053, 0.04284387, 0.30508264,
            0.77516171, 0.735498  , 0.62239045, 0.64329189, 0.26323616],
           [0.79270077, 0.28751881, 0.68060313, 0.33527429, 0.07126371,
            0.89227643, 0.74099821, 0.1413614 , 0.83575371, 0.35286316]])




```python
print(f"{random_array.min() = }")
print(f"{random_array.max() = }")
```

    random_array.min() = 0.021155988089256672
    random_array.max() = 0.9757452816746033
    

#### 14. Create a random vector of size 30 and find the mean value (★☆☆)


```python
random_vector = np.random.random((30,))
random_vector
```




    array([0.77884844, 0.074565  , 0.86744892, 0.52180949, 0.92067033,
           0.15437312, 0.5693557 , 0.27752174, 0.13960521, 0.88391854,
           0.07697968, 0.32042306, 0.01562025, 0.11639959, 0.64780853,
           0.57181685, 0.27859389, 0.55697004, 0.13823586, 0.58121217,
           0.22869387, 0.28077958, 0.12044934, 0.27486188, 0.94946611,
           0.07677409, 0.23081254, 0.52773675, 0.06160974, 0.28575782])




```python
random_vector.mean()
```




    0.38430393805805496



#### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆)


```python
random_vector = np.random.randint(low=1, high=2, size=(30,)).reshape(10,3)
random_vector[1:-1,1:-1] = 0 # this notation is super useful.
random_vector
```




    array([[1, 1, 1],
           [1, 0, 1],
           [1, 0, 1],
           [1, 0, 1],
           [1, 0, 1],
           [1, 0, 1],
           [1, 0, 1],
           [1, 0, 1],
           [1, 0, 1],
           [1, 1, 1]])



Let's do a few more for practice


```python
random_vector = np.random.randint(low=1, high=2, size=(20,)).reshape(4,5)
random_vector[1:-1,1:-1] =  "0" # funny, stringified numbers work but characters do not
# random_vector[1:-1,1:-1] = "f"
random_vector
```




    array([[1, 1, 1, 1, 1],
           [1, 0, 0, 0, 1],
           [1, 0, 0, 0, 1],
           [1, 1, 1, 1, 1]])



- Clearly, we need to have more than 2 columns because there is no middle column with just 2 columns.
- Same idea with rows, we need to have more than 2 rows because there is no middle row with just 2 rows
- Combining these ideas together, the reshape dims should be (3,3) or greater based on the input size


```python
random_vector = np.random.randint(low=1, high=2, size=(30,)).reshape(5,6)
random_vector[1:-1,1:-1] = 0 #it's nice how this part is the same for the same idea
random_vector
```




    array([[1, 1, 1, 1, 1, 1],
           [1, 0, 0, 0, 0, 1],
           [1, 0, 0, 0, 0, 1],
           [1, 0, 0, 0, 0, 1],
           [1, 1, 1, 1, 1, 1]])



#### 16. How to add a border (filled with 0's) around an existing array? (★☆☆)


```python
random_vector = np.random.randint(low=1, high=2, size=(20,)).reshape(4,5)
np.pad(random_vector, pad_width=1, mode='constant', constant_values=0)
```




    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 0, 0]])



#### 17. What is the result of the following expression? (★☆☆)
```python
0 * np.nan
np.nan == np.nan
np.inf > np.nan
np.nan - np.nan
np.nan in set([np.nan])
0.3 == 3 * 0.1
```

Should be nan, you can't multiply 0 and "not a number"


```python
0 * np.nan 
```




    nan



Should be false, one not a number need not equal another not a number


```python
np.nan == np.nan 
```




    False



Should be false, the comparison is pointless


```python
np.inf > np.nan
```




    False



nan, nan, nan


```python
np.nan - np.nan
```




    nan



true, nan is in "nan set"


```python
np.nan in set([np.nan]) 
```




    True



False, floating point precision is not exact


```python
0.3 == 3 * 0.1
```




    False




```python
3 * 0.1
```




    0.30000000000000004



#### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)

Technically this is right I guess? They don't say anything about the rest of the matrix values ;)


```python
matrix = np.tril(np.arange(1,6), k=-1)
matrix
```




    array([[0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [1, 2, 0, 0, 0],
           [1, 2, 3, 0, 0],
           [1, 2, 3, 4, 0]])



The actual answer is:


```python
matrix = np.diag(np.arange(1,5), k=-1)
matrix
```




    array([[0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0],
           [0, 2, 0, 0, 0],
           [0, 0, 3, 0, 0],
           [0, 0, 0, 4, 0]])



Just some practice:


```python
np.diag(np.arange(1,5), k=0)
```




    array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 4]])




```python
np.diag(np.arange(1,5), k=1)
```




    array([[0, 1, 0, 0, 0],
           [0, 0, 2, 0, 0],
           [0, 0, 0, 3, 0],
           [0, 0, 0, 0, 4],
           [0, 0, 0, 0, 0]])



#### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)


```python
matrix = np.zeros((8,8))
matrix[0::2,0::2] = 1
matrix[1::2,1::2] = 1
matrix
```




    array([[1., 0., 1., 0., 1., 0., 1., 0.],
           [0., 1., 0., 1., 0., 1., 0., 1.],
           [1., 0., 1., 0., 1., 0., 1., 0.],
           [0., 1., 0., 1., 0., 1., 0., 1.],
           [1., 0., 1., 0., 1., 0., 1., 0.],
           [0., 1., 0., 1., 0., 1., 0., 1.],
           [1., 0., 1., 0., 1., 0., 1., 0.],
           [0., 1., 0., 1., 0., 1., 0., 1.]])



#### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element? (★☆☆)

I had to look this up:


```python
matrix = np.random.random((6,7,8))
np.unravel_index(99, (6,7,8))
```




    (1, 5, 3)




```python
matrix[1,5,3]
```




    0.9383448132311133



#### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)

An okay solution:


```python
x = np.tile(np.array([1,0]), (1,4))
y = np.tile(np.array([0,1]), (1,4))
matrix = np.vstack((x,y))
matrix = np.tile(matrix, (4,1))
matrix
```




    array([[1, 0, 1, 0, 1, 0, 1, 0],
           [0, 1, 0, 1, 0, 1, 0, 1],
           [1, 0, 1, 0, 1, 0, 1, 0],
           [0, 1, 0, 1, 0, 1, 0, 1],
           [1, 0, 1, 0, 1, 0, 1, 0],
           [0, 1, 0, 1, 0, 1, 0, 1],
           [1, 0, 1, 0, 1, 0, 1, 0],
           [0, 1, 0, 1, 0, 1, 0, 1]])



A better solution:


```python
lego_matrix = np.array([[1,0],[0,1]])
np.tile(lego_matrix, (4,4))
```




    array([[1, 0, 1, 0, 1, 0, 1, 0],
           [0, 1, 0, 1, 0, 1, 0, 1],
           [1, 0, 1, 0, 1, 0, 1, 0],
           [0, 1, 0, 1, 0, 1, 0, 1],
           [1, 0, 1, 0, 1, 0, 1, 0],
           [0, 1, 0, 1, 0, 1, 0, 1],
           [1, 0, 1, 0, 1, 0, 1, 0],
           [0, 1, 0, 1, 0, 1, 0, 1]])



#### 22. Normalize a 5x5 random matrix (★☆☆)

Considering columns as features:


```python
matrix = np.random.random((5,5))
means = matrix.mean(axis=0)
stds = matrix.std(axis=0)
stnd_matrix = (matrix - means)/stds
stnd_matrix
```




    array([[-1.86104626,  0.89601576, -0.20306801,  1.42171603, -1.39310597],
           [ 1.02475131, -1.22849526, -0.92514335,  0.90187193,  1.50611269],
           [-0.02190145, -1.21569425, -0.80098574, -0.4950345 , -0.16300854],
           [ 0.67164855,  0.69785806,  1.85913594, -0.56179446, -0.59267962],
           [ 0.18654785,  0.85031569,  0.07006115, -1.266759  ,  0.64268143]])



considering the entire collection of 25 points as a sample  
which will have a sample mean and a sample standard deviation


```python
matrix = np.random.random((5,5))

stnd_matrix = (matrix - matrix.mean())/matrix.std()
stnd_matrix
```




    array([[ 0.97126576, -1.25866243,  1.1132708 ,  0.97516659,  1.32779957],
           [-0.75447592, -0.00851028, -0.54713304, -1.57301718,  0.10986079],
           [ 0.36812637,  1.18438609, -1.2378124 ,  0.70977168,  1.17780931],
           [-0.39664462,  1.37723218,  1.02383698, -0.60006811, -0.60826184],
           [ 0.26182278, -1.52569149, -1.76727619,  0.47042362, -0.79321902]])



#### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)

![image.png](attachment:image.png)


```python
# https://numpy.org/doc/stable/reference/arrays.dtypes.html
# https://stackoverflow.com/questions/2350072/custom-data-types-in-numpy-arrays
color = np.dtype([('r', np.ubyte), ('g', np.ubyte), ('b', np.ubyte), ('a', np.ubyte)])
```




    dtype([('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('a', 'u1')])



#### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)

What is a "real matrix product"? assuming dot product


```python
matrix1 = np.random.random((5,3))
matrix2 = np.random.random((3,2))
matrix1.dot(matrix2)
# my bad, I guess there is no cross product for matrices, it only exists for vectors
```




    array([[0.94993761, 0.64650895],
           [1.07557273, 0.7285574 ],
           [0.64810463, 0.5250493 ],
           [0.65178681, 0.48073052],
           [1.03281097, 0.82463669]])



#### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)


```python
array_1d = np.arange(0,10)
array_1d [(array_1d > 3) & (array_1d < 8)] = 0
array_1d
```




    array([0, 1, 2, 3, 0, 0, 0, 0, 8, 9])



#### 26. What is the output of the following script? (★☆☆)
```python
# Author: Jake VanderPlas

print(sum(range(5),-1))
from numpy import *
print(sum(range(5),-1))
```

should add 0,1,2,3,4 and then subtract 1  
i.e. 0+1+2+3+4 - 1 = 10 - 1 = 9


```python
sum(range(5),-1)
```




    10



performs the sum along the last axis of the input  
i.e. sums all elements in the array


```python
from numpy import *
sum(range(5),-1)

```




    10



#### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)
```python
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
```


```python
z = np.random.randint(low=0,high=10,size=(10,))
z
```




    array([1, 1, 3, 0, 9, 7, 2, 6, 8, 6])



I don't know the answer, so let's try it out


```python
z**z
```




    array([        1,         1,        27,         1, 387420489,    823543,
                   4,     46656,  16777216,     46656])



I think this is done elementwise.  
But isn't it weird that 0**0 is 1?


```python
2 << z >> 2
```




    array([  1,   1,   4,   0, 256,  64,   2,  32, 128,  32], dtype=int32)



Funny how we don't get the original array after  
left shifting and right shifting by the same amount


```python
z < -z

```




    array([False, False, False, False, False, False, False, False, False,
           False])



Well duh  
If you're worried about the 0  
It's fine because the condition says less than  
not less than or equal to


```python
z
```




    array([1, 1, 3, 0, 9, 7, 2, 6, 8, 6])




```python
1j*z
# one Jay Z ? ;)
```




    array([0.+1.j, 0.+1.j, 0.+3.j, 0.+0.j, 0.+9.j, 0.+7.j, 0.+2.j, 0.+6.j,
           0.+8.j, 0.+6.j])




```python
z  / 1 / 1
```




    array([1., 1., 3., 0., 9., 7., 2., 6., 8., 6.])



Funny, converts the array to floats  


```python
z < z > z
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    c:\Users\kaush\Desktop\python_projects_new\gitgud-with-numpy\100_Numpy_exercises_etrama.ipynb Cell 96 line 1
    ----> <a href='vscode-notebook-cell:/c%3A/Users/kaush/Desktop/python_projects_new/gitgud-with-numpy/100_Numpy_exercises_etrama.ipynb#Y516sZmlsZQ%3D%3D?line=0'>1</a> z < z > z
    

    ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()



```python
z < z
```




    array([False, False, False, False, False, False, False, False, False,
           False])




```python
z > z
```




    array([False, False, False, False, False, False, False, False, False,
           False])



#### 28. What are the result of the following expressions? (★☆☆)
```python
np.array(0) / np.array(0)
np.array(0) // np.array(0)
np.array([np.nan]).astype(int).astype(float)
```

I don't know, so let's find out


```python
np.array(0) / np.array(0)
```

    C:\Users\kaush\AppData\Local\Temp\ipykernel_27348\2670511084.py:2: RuntimeWarning: invalid value encountered in divide
      np.array(0) / np.array(0)
    




    nan




```python
np.array(0) // np.array(0)
```

    C:\Users\kaush\AppData\Local\Temp\ipykernel_27348\2018018105.py:1: RuntimeWarning: divide by zero encountered in floor_divide
      np.array(0) // np.array(0)
    




    0




```python
np.array([np.nan]).astype(int).astype(float)
```

    C:\Users\kaush\AppData\Local\Temp\ipykernel_27348\699728972.py:1: RuntimeWarning: invalid value encountered in cast
      np.array([np.nan]).astype(int).astype(float)
    




    array([-2.14748365e+09])



Say what now?


```python
np.array([nan])
```




    array([nan])




```python
np.array([nan]).astype(int)
```

    C:\Users\kaush\AppData\Local\Temp\ipykernel_27348\3805164281.py:1: RuntimeWarning: invalid value encountered in cast
      np.array([nan]).astype(int)
    




    array([-2147483648])




```python
np.array([nan]).astype(int).astype(float)
```

    C:\Users\kaush\AppData\Local\Temp\ipykernel_27348\2776176757.py:1: RuntimeWarning: invalid value encountered in cast
      np.array([nan]).astype(int).astype(float)
    




    array([-2.14748365e+09])



I guess nan is initialized as a random LARGE number.

The questions are getting tougher from this point onwards, so just assume that everything was done with the help our friend Google.

#### 29. How to round away from zero a float array ? (★☆☆)

I had no idea, what this meant and had to check Rougier's solution:


```python
Z = np.random.uniform(-10,+10,10)
Z
```




    array([ 3.78000333,  9.18521434, -2.8426338 , -3.62545325, -0.35721967,
            5.68759169,  9.06396618, -3.77583255, -2.34084291, -0.08756355])




```python
print(np.copysign(np.ceil(np.abs(Z)), Z))
```

    [ 4. 10. -3. -4. -1.  6. 10. -4. -3. -1.]
    


```python
# More readable but less efficient
print(np.where(Z>0, np.ceil(Z), np.floor(Z)))
```

    [ 4. 10. -3. -4. -1.  6. 10. -4. -3. -1.]
    

#### 30. How to find common values between two arrays? (★☆☆)

Typically, I would do this:


```python
array1 = np.arange(0,10)
array2 = np.arange(5,15)
```


```python
array1
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
array2
```




    array([ 5,  6,  7,  8,  9, 10, 11, 12, 13, 14])




```python
set(array1).intersection(set(array2))
```




    {5, 6, 7, 8, 9}



But, I guess there is a more numpy way to do it, which we'll look up.


```python
np.intersect1d(array1, array2)
```




    array([5, 6, 7, 8, 9])



The arrays are treated as "flat arrays" to find common elements if they are not already 1D


```python
matrix1 = np.array([[1,2],[3,4]])
matrix2 = np.array([[3,4],[5,6]])
np.intersect1d(matrix1, matrix2)
```




    array([3, 4])



#### 31. How to ignore all numpy warnings (not recommended)? (★☆☆)

Typically, I would do this


```python
import warnings
warnings.filterwarnings('ignore')
```

No more divide by zero warnings:


```python
np.array(0) / np.array(0)
```




    nan



The more numpy way is, Rougier's solutions themsevles have the best answer:


```python
# Author: Rougier
# Suicide mode on
defaults = np.seterr(all="ignore")
Z = np.ones(1) / 0

# Back to sanity
_ = np.seterr(**defaults)
```


```python
# Equivalently with a context manager
with np.errstate(all="ignore"):
    np.arange(3) / 0
```

#### 32. Is the following expressions true? (★☆☆)
```python
np.sqrt(-1) == np.emath.sqrt(-1)
```

What even is emath? Apparently, it finds the complex square root.  
Idk, if sqrt can handle square roots of negative numbers, I would guess not.  
Otherwise, why would we have emath?  
Expression should be false / lead to an error.  


```python
np.sqrt(-1) == np.emath.sqrt(-1)
```




    False




```python
np.emath.sqrt(-1)
```




    1j




```python
np.sqrt(-1)
```




    nan



#### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆)

This I had to look up.


```python
today = np.datetime64("today")
yesterday = today - np.timedelta64(1)
tomorrow = today + np.timedelta64(1)
```


```python
yesterday, today, tomorrow
```




    (numpy.datetime64('2023-10-24'),
     numpy.datetime64('2023-10-25'),
     numpy.datetime64('2023-10-26'))



#### 34. How to get all the dates corresponding to the month of July 2016? (★★☆)


```python
# https://www.geeksforgeeks.org/display-all-the-dates-for-a-particular-month-using-numpy/
np.arange("2016-07","2016-08",dtype='datetime64[D]')
```




    array(['2016-07-01', '2016-07-02', '2016-07-03', '2016-07-04',
           '2016-07-05', '2016-07-06', '2016-07-07', '2016-07-08',
           '2016-07-09', '2016-07-10', '2016-07-11', '2016-07-12',
           '2016-07-13', '2016-07-14', '2016-07-15', '2016-07-16',
           '2016-07-17', '2016-07-18', '2016-07-19', '2016-07-20',
           '2016-07-21', '2016-07-22', '2016-07-23', '2016-07-24',
           '2016-07-25', '2016-07-26', '2016-07-27', '2016-07-28',
           '2016-07-29', '2016-07-30', '2016-07-31'], dtype='datetime64[D]')



#### 35. How to compute ((A+B)*(-A/2)) in place (without copy)? (★★☆)

If we write the equation like we have below, it seems quite trivial to do this. Am I missing something?

$$
 (\mathrm{A} + \mathrm{B})(-\mathrm{A}/2)
$$


```python
A = np.random.random((10,10))
B = np.random.random((10,10))
np.dot(A+B, -A/2)
```




    array([[-2.3345593 , -2.5181915 , -2.05993547, -1.51700525, -2.11205926,
            -1.8704557 , -2.82968387, -2.05317614, -1.78929567, -2.41079191],
           [-3.27060258, -3.13411811, -2.37829474, -1.9536335 , -2.46305948,
            -2.82093294, -3.36122801, -2.60408213, -2.39491811, -2.66310532],
           [-3.14721551, -3.02074321, -2.46912062, -2.08869699, -2.8502098 ,
            -2.67210798, -3.81826711, -2.94231102, -2.47939385, -2.99949339],
           [-3.06704252, -2.98084217, -2.47617424, -1.95498037, -2.42177667,
            -2.74786496, -3.90393746, -2.50734506, -2.41337778, -2.67133232],
           [-3.04055681, -2.95362743, -2.29415802, -2.3134393 , -2.43457362,
            -2.83238168, -4.08648297, -2.43740852, -2.83274016, -2.78154982],
           [-2.97949928, -2.95424164, -2.29670243, -2.33068081, -2.74954399,
            -2.59747968, -3.59870839, -1.9857586 , -2.42700608, -2.95020124],
           [-2.70521905, -2.59741715, -2.1642417 , -1.9424324 , -2.22679537,
            -2.41211133, -3.54359569, -1.90794574, -2.65796665, -2.557535  ],
           [-3.47187939, -3.10711104, -2.37754957, -2.26911595, -2.76189941,
            -2.67650432, -3.34597961, -3.07979332, -2.50241298, -2.83714537],
           [-3.35136982, -2.60855687, -2.29513495, -1.9495123 , -2.70112929,
            -3.05337062, -3.38965928, -2.18342522, -2.66239322, -2.34052408],
           [-2.03017891, -1.93608522, -1.52731301, -1.33161686, -1.67883936,
            -1.73416623, -2.4997938 , -1.77631822, -1.55552652, -1.98359694]])



I completely missed the point. The keyword is to do it "in place".


```python
np.dot(A+B, -A/2,out=A)
```




    array([[-2.79687526, -1.84248069, -2.70266152, -1.93581225, -2.66175009,
            -2.48463918, -2.94236796, -3.42826331, -3.61348636, -2.90494294],
           [-2.56621211, -1.82469589, -2.66723905, -1.63511762, -2.88760047,
            -2.31244914, -2.96539791, -2.94692464, -3.14698994, -2.56890546],
           [-2.38858593, -1.35370013, -2.18440092, -1.47466475, -2.44465977,
            -1.86424866, -2.6297063 , -2.58292364, -2.62069968, -2.36782019],
           [-2.45679257, -1.38991901, -2.13033762, -1.98592295, -2.61107653,
            -2.13516764, -2.43723036, -2.50601267, -2.85202388, -2.03948006],
           [-2.30954782, -1.44425766, -1.97984039, -1.60472065, -2.57214731,
            -2.01744569, -2.36872234, -2.4916194 , -2.71303063, -1.8656976 ],
           [-3.1877976 , -2.27418332, -3.0426209 , -2.36296517, -3.65272166,
            -2.77775881, -3.80333857, -3.8091126 , -3.8999394 , -2.83813297],
           [-2.08402791, -1.35533869, -1.901799  , -1.91015726, -2.5227628 ,
            -1.56173424, -2.43114624, -2.35814187, -2.57606471, -1.72567408],
           [-2.36862586, -1.57408583, -2.08053001, -1.71915793, -2.45262444,
            -2.21663944, -2.31592278, -2.62285418, -2.65237471, -1.91134162],
           [-3.02457789, -1.82978335, -2.75520291, -1.96818795, -3.30956181,
            -2.68465279, -3.25999162, -3.37276716, -3.90885561, -3.00682314],
           [-2.8032067 , -1.80675776, -2.45282667, -1.82085936, -2.69690201,
            -2.60722972, -2.60853734, -3.39637856, -3.52134065, -2.63616216]])



So we need to use the out keyword and not the assignment operator.


```python
A
```




    array([[-2.79687526, -1.84248069, -2.70266152, -1.93581225, -2.66175009,
            -2.48463918, -2.94236796, -3.42826331, -3.61348636, -2.90494294],
           [-2.56621211, -1.82469589, -2.66723905, -1.63511762, -2.88760047,
            -2.31244914, -2.96539791, -2.94692464, -3.14698994, -2.56890546],
           [-2.38858593, -1.35370013, -2.18440092, -1.47466475, -2.44465977,
            -1.86424866, -2.6297063 , -2.58292364, -2.62069968, -2.36782019],
           [-2.45679257, -1.38991901, -2.13033762, -1.98592295, -2.61107653,
            -2.13516764, -2.43723036, -2.50601267, -2.85202388, -2.03948006],
           [-2.30954782, -1.44425766, -1.97984039, -1.60472065, -2.57214731,
            -2.01744569, -2.36872234, -2.4916194 , -2.71303063, -1.8656976 ],
           [-3.1877976 , -2.27418332, -3.0426209 , -2.36296517, -3.65272166,
            -2.77775881, -3.80333857, -3.8091126 , -3.8999394 , -2.83813297],
           [-2.08402791, -1.35533869, -1.901799  , -1.91015726, -2.5227628 ,
            -1.56173424, -2.43114624, -2.35814187, -2.57606471, -1.72567408],
           [-2.36862586, -1.57408583, -2.08053001, -1.71915793, -2.45262444,
            -2.21663944, -2.31592278, -2.62285418, -2.65237471, -1.91134162],
           [-3.02457789, -1.82978335, -2.75520291, -1.96818795, -3.30956181,
            -2.68465279, -3.25999162, -3.37276716, -3.90885561, -3.00682314],
           [-2.8032067 , -1.80675776, -2.45282667, -1.82085936, -2.69690201,
            -2.60722972, -2.60853734, -3.39637856, -3.52134065, -2.63616216]])



Rougier's solution is kinda neat too:


```python
A = np.random.random((10,10))
B = np.random.random((10,10))
np.add(A,B,out=B)
np.divide(A,2,out=A)
np.negative(A,out=A)
np.multiply(B,A)
```




    array([[-3.29048037, -0.78878275, -3.45398628, -1.45232849, -3.50597586,
            -2.52302377, -3.49792305, -5.10535362, -4.97389611, -2.90221083],
           [-2.22054238, -1.11667138, -3.42500069, -0.68487702, -3.75151033,
            -1.8603307 , -4.29460978, -2.96714799, -4.68559821, -3.20089576],
           [-2.37429049, -0.41416725, -2.24417686, -0.53475374, -2.48534547,
            -1.6899615 , -3.31570457, -2.0553249 , -3.40262979, -1.92397295],
           [-2.67700134, -0.88968944, -1.20508543, -1.6080221 , -2.70996818,
            -2.02033007, -2.95596772, -2.47212201, -3.45588847, -1.92408554],
           [-2.21071886, -0.86786971, -1.63669418, -1.22240397, -2.47724667,
            -1.26584674, -2.65416758, -2.16602191, -2.65610688, -1.56085344],
           [-4.34504904, -1.7142312 , -3.95018325, -1.73114116, -5.31761119,
            -3.58803598, -6.44673159, -5.80161508, -5.7827312 , -3.75268081],
           [-1.67052859, -0.42384923, -0.88832529, -1.21499156, -2.79377543,
            -0.90656774, -2.41320961, -2.58971498, -2.97469785, -1.17543946],
           [-2.43596681, -1.19526634, -1.94914164, -1.16963834, -1.96583522,
            -2.34140668, -2.58842985, -2.23631124, -3.0632712 , -1.57736341],
           [-3.09378713, -1.53303782, -2.66677391, -1.15440958, -5.3741457 ,
            -3.31382941, -4.29875552, -4.52351393, -6.33462046, -3.310913  ],
           [-3.3175497 , -0.99323556, -2.52284929, -1.55911814, -3.22324397,
            -2.29484295, -2.20763428, -5.51465628, -4.75375221, -2.25053305]])



#### 36. Extract the integer part of a random array of positive numbers using 4 different methods (★★☆)


```python
a = np.random.uniform(low=0,high=1.0,size=(10))*100
```

The array "a" contains random positive integers.

Method 1:


```python
a
```




    array([22.11421753, 51.86133863, 18.51164469, 58.50565343, 68.14403282,
           48.08201432, 28.72401268, 38.50121503, 83.77582821, 80.23137176])




```python
a.astype(int)
```




    array([22, 51, 18, 58, 68, 48, 28, 38, 83, 80])



Method 2:


```python
# https://stackoverflow.com/questions/6681743/splitting-a-number-into-the-integer-and-decimal-parts
a-a%1
```




    array([22., 51., 18., 58., 68., 48., 28., 38., 83., 80.])



Method 3:


```python
# this is venturing out of numpy but oh well
# https://stackoverflow.com/questions/6681743/splitting-a-number-into-the-integer-and-decimal-parts
list_a = a.tolist()
for idx, val in enumerate(list_a):
    list_a[idx] = str(val).split('.')[0]
np.array(list_a)
```




    array(['22', '51', '18', '58', '68', '48', '28', '38', '83', '80'],
          dtype='<U2')



Method 4:


```python
a // 1
```




    array([22., 51., 18., 58., 68., 48., 28., 38., 83., 80.])



Thanks to Rougier, we hav a couple more:

Method 5:


```python
np.floor(a)
```




    array([22., 51., 18., 58., 68., 48., 28., 38., 83., 80.])



Method 6:


```python
np.trunc(a)
```




    array([22., 51., 18., 58., 68., 48., 28., 38., 83., 80.])



#### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)


```python
vector = np.arange(0,5)
np.vstack((vector, vector, vector, vector, vector))
```




    array([[0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4]])



Rougier's solution is again more elegant:


```python
Z = np.zeros((5,5))
Z += np.arange(5)
Z
```




    array([[0., 1., 2., 3., 4.],
           [0., 1., 2., 3., 4.],
           [0., 1., 2., 3., 4.],
           [0., 1., 2., 3., 4.],
           [0., 1., 2., 3., 4.]])



Another elegant solution by Rougier using tile:


```python
np.tile(np.arange(0,5),(5,1))
```




    array([[0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4],
           [0, 1, 2, 3, 4]])



#### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)

A generator function called "jenny":


```python
def jenny():
    for i in range(1,11):
        yield i
```


```python
for i in jenny():
    print(i)
```

    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    


```python
a = np.array([])
for i in jenny():
    a  = np.hstack((a,i))
a
```




    array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])



Yet again, Rougier delivers:


```python
np.fromiter(jenny(), dtype=float, count=-1)
# count -1 means that we use all the values from the iterator
```




    array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])



It seems like I need to think less mechanically and more "numpy"ly.

#### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)


```python
np.linspace(0,1,12)[1:-1]
```




    array([0.09090909, 0.18181818, 0.27272727, 0.36363636, 0.45454545,
           0.54545455, 0.63636364, 0.72727273, 0.81818182, 0.90909091])



Rougier:


```python
np.linspace(0,1,11,endpoint=False)[1:]
```




    array([0.09090909, 0.18181818, 0.27272727, 0.36363636, 0.45454545,
           0.54545455, 0.63636364, 0.72727273, 0.81818182, 0.90909091])



#### 40. Create a random vector of size 10 and sort it (★★☆)


```python
random_vector = np.random.randint(low=0, high=10, size=(10,))
random_vector
```




    array([3, 6, 5, 3, 8, 8, 1, 6, 9, 0])



    Not in place:


```python
sorted(random_vector)
```




    [0, 1, 3, 3, 5, 6, 6, 8, 8, 9]



In place:


```python
random_vector.sort()
random_vector
```




    array([0, 1, 3, 3, 5, 6, 6, 8, 8, 9])



#### 41. How to sum a small array faster than np.sum? (★★☆)

No idea. Looking up.


```python
#https://stackoverflow.com/questions/10922231/pythons-sum-vs-numpys-numpy-sum
array = np.arange(0,10)
```

Sum is supposed to be a tiny bit faster for small arrays.

![image.png](attachment:image.png)


```python
%time
np.sum(array)
```

    CPU times: total: 0 ns
    Wall time: 0 ns
    




    45




```python
%time
sum(array)
```

    CPU times: total: 0 ns
    Wall time: 0 ns
    




    45



The code was not able to capture the time difference.


```python
import time
time1 = time.time()
np.sum(array)
time2 = time.time()
print(f"Time taken: {time2-time1}")
```

    Time taken: 0.0010027885437011719
    


```python
import time
time1 = time.time()
sum(array)
time2 = time.time()
print(f"Time taken: {time2-time1}")
```

    Time taken: 0.0
    

To be rigorous, we should probably run the same code like 10 times and see if sum is consistently faster than np.sum() for small arrays. But this is good enough for me.

#### 42. Consider two random array A and B, check if they are equal (★★☆)


```python
A = np.random.random((10,10))
B = np.random.random((10,10))
np.all(A==B)
```




    False




```python
A = np.random.randint(low=0, high=10, size=(10,10))
B = A.copy()
np.all(A==B)
```




    True




```python
B[0,0] = 100
np.all(A==B)
```




    False



Rougier's solution is more numpy like:


```python
A = np.random.randint(low=0, high=10, size=(10,10))
B = A.copy()
np.allclose(A,B)
```




    True




```python
np.array_equal(A,B)
```




    True



#### 43. Make an array immutable (read-only) (★★☆)


```python
# https://stackoverflow.com/questions/5541324/immutable-numpy-array
a = np.random.random((5,5))
a
```




    array([[0.8227012 , 0.12601625, 0.42540863, 0.97627089, 0.63388622],
           [0.75687311, 0.45598492, 0.10625496, 0.58104675, 0.09436362],
           [0.16486368, 0.13800275, 0.21366973, 0.76358301, 0.52054695],
           [0.08156548, 0.09853984, 0.17364485, 0.00972851, 0.74932949],
           [0.4892877 , 0.273489  , 0.73034423, 0.20120066, 0.16257508]])




```python
a.setflags(write=False)
```


```python
a[0, 0] = 100
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    c:\Users\kaush\Desktop\python_projects_new\gitgud-with-numpy\100_Numpy_exercises_etrama.ipynb Cell 243 line 1
    ----> <a href='vscode-notebook-cell:/c%3A/Users/kaush/Desktop/python_projects_new/gitgud-with-numpy/100_Numpy_exercises_etrama.ipynb#Z1016sZmlsZQ%3D%3D?line=0'>1</a> a[0, 0] = 100
    

    ValueError: assignment destination is read-only


#### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)


```python
cartesian_coords = np.random.randint(low=0, high=10, size=(10,2))
cartesian_coords
```




    array([[9, 5],
           [6, 0],
           [9, 7],
           [0, 1],
           [3, 2],
           [7, 6],
           [5, 5],
           [6, 1],
           [5, 4],
           [5, 1]])



Cartesian to polar coordinates:
$$
r = \sqrt{x^2 + y^2} 
$$
$$
\theta = \arctan(y/x)
$$


```python
#https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
def polar(x, y) -> tuple:
  """returns rho, theta (degrees)"""
  return np.hypot(x, y), np.degrees(np.arctan2(y, x))
```


```python
polar(3,4)
```




    (5.0, 53.13010235415598)




```python
polar(cartesian_coords[0][0], cartesian_coords[0][1])
```




    (10.295630140987, 29.054604099077146)




```python
polar(*cartesian_coords[0])
```




    (10.295630140987, 29.054604099077146)




```python
polar(cartesian_coords[:,0], cartesian_coords[:,1])
```




    (array([10.29563014,  6.        , 11.40175425,  1.        ,  3.60555128,
             9.21954446,  7.07106781,  6.08276253,  6.40312424,  5.09901951]),
     array([29.0546041 ,  0.        , 37.87498365, 90.        , 33.69006753,
            40.60129465, 45.        ,  9.46232221, 38.65980825, 11.30993247]))



#### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)


```python
random_vector = np.random.random((10,))
random_vector
```




    array([0.74971807, 0.90937433, 0.63791424, 0.07883616, 0.92606344,
           0.57586556, 0.84885824, 0.64565938, 0.85861547, 0.29818429])




```python
np.max(random_vector), np.argmax(random_vector)
```




    (0.9260634367924555, 4)




```python
random_vector[np.argmax(random_vector)] = 0
random_vector
```




    array([0.74971807, 0.90937433, 0.63791424, 0.07883616, 0.        ,
           0.57586556, 0.84885824, 0.64565938, 0.85861547, 0.29818429])



#### 46. Create a structured array with `x` and `y` coordinates covering the [0,1]x[0,1] area (★★☆)


```python
#https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
x = np.linspace(0,1,10)
y = np.linspace(0,1,10)
np.meshgrid(x, y)
```




    [array([[0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
             0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ],
            [0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
             0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ],
            [0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
             0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ],
            [0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
             0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ],
            [0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
             0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ],
            [0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
             0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ],
            [0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
             0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ],
            [0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
             0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ],
            [0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
             0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ],
            [0.        , 0.11111111, 0.22222222, 0.33333333, 0.44444444,
             0.55555556, 0.66666667, 0.77777778, 0.88888889, 1.        ]]),
     array([[0.        , 0.        , 0.        , 0.        , 0.        ,
             0.        , 0.        , 0.        , 0.        , 0.        ],
            [0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111,
             0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111],
            [0.22222222, 0.22222222, 0.22222222, 0.22222222, 0.22222222,
             0.22222222, 0.22222222, 0.22222222, 0.22222222, 0.22222222],
            [0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333,
             0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333],
            [0.44444444, 0.44444444, 0.44444444, 0.44444444, 0.44444444,
             0.44444444, 0.44444444, 0.44444444, 0.44444444, 0.44444444],
            [0.55555556, 0.55555556, 0.55555556, 0.55555556, 0.55555556,
             0.55555556, 0.55555556, 0.55555556, 0.55555556, 0.55555556],
            [0.66666667, 0.66666667, 0.66666667, 0.66666667, 0.66666667,
             0.66666667, 0.66666667, 0.66666667, 0.66666667, 0.66666667],
            [0.77777778, 0.77777778, 0.77777778, 0.77777778, 0.77777778,
             0.77777778, 0.77777778, 0.77777778, 0.77777778, 0.77777778],
            [0.88888889, 0.88888889, 0.88888889, 0.88888889, 0.88888889,
             0.88888889, 0.88888889, 0.88888889, 0.88888889, 0.88888889],
            [1.        , 1.        , 1.        , 1.        , 1.        ,
             1.        , 1.        , 1.        , 1.        , 1.        ]])]



Rougier demonstrates a way with custom dtypes:


```python
Z = np.zeros((5,5), [('x',float),('y',float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,5),
                             np.linspace(0,1,5))
Z
```




    array([[(0.  , 0.  ), (0.25, 0.  ), (0.5 , 0.  ), (0.75, 0.  ),
            (1.  , 0.  )],
           [(0.  , 0.25), (0.25, 0.25), (0.5 , 0.25), (0.75, 0.25),
            (1.  , 0.25)],
           [(0.  , 0.5 ), (0.25, 0.5 ), (0.5 , 0.5 ), (0.75, 0.5 ),
            (1.  , 0.5 )],
           [(0.  , 0.75), (0.25, 0.75), (0.5 , 0.75), (0.75, 0.75),
            (1.  , 0.75)],
           [(0.  , 1.  ), (0.25, 1.  ), (0.5 , 1.  ), (0.75, 1.  ),
            (1.  , 1.  )]], dtype=[('x', '<f8'), ('y', '<f8')])



#### 47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj)) (★★☆)

From Wikipedia:

![image.png](attachment:image.png)

I had an idea about how to do this but it was completely wrong. So here's an analysis of Rougier's solution:


```python
X = np.arange(8)
```


```python
Y = X + 0.5
```

Some [docs](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.outer.html) on np.ufunc.outer:

![image.png](attachment:image.png)


```python
C = 1.0 / np.subtract.outer(X, Y)
C
```




    array([[-2.        , -0.66666667, -0.4       , -0.28571429, -0.22222222,
            -0.18181818, -0.15384615, -0.13333333],
           [ 2.        , -2.        , -0.66666667, -0.4       , -0.28571429,
            -0.22222222, -0.18181818, -0.15384615],
           [ 0.66666667,  2.        , -2.        , -0.66666667, -0.4       ,
            -0.28571429, -0.22222222, -0.18181818],
           [ 0.4       ,  0.66666667,  2.        , -2.        , -0.66666667,
            -0.4       , -0.28571429, -0.22222222],
           [ 0.28571429,  0.4       ,  0.66666667,  2.        , -2.        ,
            -0.66666667, -0.4       , -0.28571429],
           [ 0.22222222,  0.28571429,  0.4       ,  0.66666667,  2.        ,
            -2.        , -0.66666667, -0.4       ],
           [ 0.18181818,  0.22222222,  0.28571429,  0.4       ,  0.66666667,
             2.        , -2.        , -0.66666667],
           [ 0.15384615,  0.18181818,  0.22222222,  0.28571429,  0.4       ,
             0.66666667,  2.        , -2.        ]])




```python
np.linalg.det(C)
```




    3638.1636371179666



#### 48. Print the minimum and maximum representable value for each numpy scalar type (★★☆)


```python
#https://numpy.org/doc/stable/reference/arrays.scalars.html
#https://stackoverflow.com/questions/21968643/what-is-a-scalar-in-numpy
np.ScalarType
```




    (int,
     float,
     complex,
     bool,
     bytes,
     str,
     memoryview,
     numpy.bool_,
     numpy.complex64,
     numpy.complex128,
     numpy.clongdouble,
     numpy.float16,
     numpy.float32,
     numpy.float64,
     numpy.longdouble,
     numpy.int8,
     numpy.int16,
     numpy.int32,
     numpy.intc,
     numpy.int64,
     numpy.datetime64,
     numpy.timedelta64,
     numpy.object_,
     numpy.bytes_,
     numpy.str_,
     numpy.uint8,
     numpy.uint16,
     numpy.uint32,
     numpy.uintc,
     numpy.uint64,
     numpy.void)




```python
# https://stackoverflow.com/questions/21968643/what-is-a-scalar-in-numpy
np.iinfo(int).min
```




    -2147483648




```python
np.iinfo(int).max
```




    2147483647



Making this a little more readable:


```python
# https://stackoverflow.com/questions/1823058/how-to-print-a-number-using-commas-as-thousands-separators
f"{np.iinfo(int).min:,}"
```




    '-2,147,483,648'




```python
f"{np.iinfo(int).max:,}"
```




    '2,147,483,647'



Rougier's solution also mentions finfo, which has the same usage as iinfo.
iinfo : integer info
finfo : floating info
The iinfo function won't work on floating points and vice-versa.

Net-net, use the iinfo function along with min and max attributes.

#### 49. How to print all the values of an array? (★★☆)


```python
x = np.random.random((50,50))
x
```




    array([[9.09493927e-01, 2.39865472e-02, 2.55512699e-01, 5.86328195e-01,
            7.11310838e-02, 3.43906434e-01, 1.55961957e-02, 4.27471885e-01,
            7.55280892e-01, 8.91569347e-01, 6.85674566e-01, 3.53757164e-01,
            1.10756416e-01, 2.19419896e-01, 1.48348921e-01, 8.86062128e-01,
            3.29159187e-01, 3.42868380e-02, 7.20196429e-01, 9.40518704e-01,
            1.91526779e-01, 5.78030490e-01, 9.09834624e-01, 3.31523512e-01,
            6.25675801e-01, 8.29793218e-01, 1.98280128e-01, 4.94173211e-01,
            8.84756044e-01, 5.16241643e-01, 4.12229321e-01, 2.69868841e-02,
            2.62166565e-01, 8.43344805e-01, 5.28991601e-01, 3.51315584e-01,
            6.04624653e-01, 6.07264748e-01, 9.48114311e-01, 4.78166438e-01,
            6.68048594e-01, 8.59508647e-01, 2.65764461e-01, 9.68481541e-01,
            1.93221498e-01, 4.45372291e-01, 9.15174093e-01, 2.46202160e-01,
            5.89698895e-01, 3.21986029e-02],
           [8.77345607e-01, 6.58711207e-01, 9.02516252e-01, 9.87827993e-01,
            5.40404639e-01, 4.68443738e-01, 1.78858010e-01, 8.66190209e-01,
            5.84171129e-01, 7.28585681e-01, 5.76629159e-01, 6.28880944e-01,
            8.99590495e-01, 2.58480230e-01, 4.28828180e-01, 3.25574100e-01,
            3.88793491e-01, 9.47620604e-01, 5.96951521e-01, 4.44672279e-01,
            5.51454680e-01, 1.15206288e-01, 1.42543001e-01, 6.21700899e-01,
            9.79686413e-01, 2.67082399e-02, 3.32875674e-01, 7.29076235e-01,
            4.06204298e-04, 4.80769468e-03, 7.68993273e-01, 3.02584403e-01,
            1.48593816e-01, 9.40445524e-01, 1.86897046e-01, 1.21953422e-01,
            6.03991981e-01, 5.03449853e-01, 2.76685087e-01, 3.65547826e-01,
            4.24615207e-01, 7.63027836e-01, 6.68590876e-01, 3.47521237e-01,
            7.83400712e-01, 9.72431596e-02, 7.75169006e-01, 1.08304787e-01,
            1.70414332e-01, 9.35440419e-01],
           [9.00761296e-01, 4.24074718e-01, 4.57967226e-02, 6.69926822e-01,
            8.13062571e-02, 4.24992824e-01, 1.57816435e-01, 4.49856267e-01,
            8.67017093e-01, 5.85894195e-01, 5.58185041e-01, 3.79443337e-01,
            3.06991103e-01, 9.13087437e-01, 8.53450577e-01, 1.14077617e-01,
            2.69261506e-01, 2.44527850e-01, 5.62002921e-01, 3.21048162e-01,
            7.03234477e-01, 6.96583530e-01, 7.00936075e-01, 2.86291409e-01,
            3.47557687e-01, 7.36594848e-01, 1.53182815e-01, 4.08231157e-01,
            4.05792766e-02, 6.13423617e-01, 2.29961660e-01, 9.56343702e-01,
            8.88213692e-01, 3.07395235e-01, 9.69745029e-01, 8.09560031e-01,
            8.68280755e-01, 7.60896385e-01, 4.66355383e-01, 5.35764485e-01,
            4.06991140e-01, 2.40143821e-01, 7.49911318e-01, 6.10924384e-01,
            5.97709273e-01, 7.66935214e-01, 1.68155516e-02, 5.32416390e-01,
            3.97378127e-01, 1.58381915e-01],
           [5.85832676e-01, 4.69261970e-02, 4.49434847e-01, 3.93760601e-01,
            7.89355117e-01, 3.53199694e-01, 5.72347779e-01, 6.01739135e-01,
            1.67563696e-01, 3.99907027e-01, 5.45776104e-01, 6.30217635e-01,
            1.30219306e-01, 9.07390217e-01, 7.91456584e-01, 3.44962215e-01,
            6.70328327e-01, 1.27873375e-02, 1.19891384e-01, 4.08509191e-01,
            9.61505654e-01, 2.06156903e-01, 7.16620516e-01, 7.87112304e-01,
            4.07001026e-01, 8.01709504e-02, 1.35851157e-01, 3.59771384e-01,
            4.19368031e-01, 1.76208130e-01, 5.47808353e-01, 8.94358692e-01,
            9.68440325e-01, 5.96414556e-02, 6.53637163e-01, 8.10309288e-01,
            9.82330863e-01, 7.75340636e-01, 6.87218708e-01, 5.58430316e-01,
            3.65192408e-01, 3.48994339e-01, 6.80945362e-01, 1.93069374e-01,
            1.18215972e-01, 1.08832242e-01, 1.43459333e-01, 5.69379925e-01,
            1.56327091e-01, 1.21767518e-01],
           [2.24502844e-01, 9.48486015e-01, 3.32807718e-01, 9.81136468e-01,
            4.63648396e-01, 8.30050425e-01, 4.72812349e-01, 7.52662178e-01,
            3.69170471e-01, 2.40943428e-01, 9.36711205e-01, 3.55939723e-01,
            4.91873493e-01, 1.42478978e-01, 5.34181068e-01, 1.28885039e-01,
            3.65678160e-01, 7.23522758e-01, 6.75134163e-01, 5.77673421e-01,
            5.44011638e-01, 5.06741244e-01, 4.60582211e-01, 9.49955718e-01,
            1.82737754e-01, 1.88716047e-02, 1.70006544e-01, 2.63572285e-01,
            5.20921547e-01, 3.38468366e-01, 5.44394611e-01, 6.92905927e-01,
            4.58719679e-01, 2.46288435e-01, 8.70814439e-01, 8.87593144e-01,
            6.24917117e-02, 7.18742286e-01, 3.51460287e-02, 9.89762660e-01,
            1.67179022e-01, 9.05753274e-01, 7.47057685e-01, 9.68457248e-01,
            1.80235184e-01, 5.53476121e-01, 1.56274503e-01, 1.53031705e-01,
            7.77363834e-01, 3.57955720e-01],
           [6.44103448e-01, 6.10652955e-01, 8.61528482e-01, 1.86192132e-01,
            7.57056675e-01, 6.25917972e-03, 8.88997645e-01, 5.60190295e-01,
            7.78102085e-01, 7.68879259e-01, 5.08866976e-01, 4.14625899e-01,
            7.45531934e-01, 9.42978416e-01, 5.15441545e-02, 5.07498738e-01,
            6.22196385e-01, 1.31233597e-01, 1.77969975e-01, 1.76071555e-04,
            3.05662794e-01, 7.32209704e-01, 8.23953013e-01, 4.67618791e-01,
            5.97785255e-01, 5.16311088e-01, 7.33762585e-02, 1.01120547e-01,
            2.68163995e-01, 5.35862790e-01, 7.80980015e-01, 6.01414149e-01,
            5.43838023e-02, 8.22605155e-02, 2.86067059e-01, 3.22367166e-01,
            2.34773916e-01, 5.81265042e-01, 2.83901549e-01, 8.08501149e-01,
            2.50157001e-01, 3.98508576e-02, 4.29980951e-01, 1.94903927e-01,
            8.65610184e-01, 7.02643108e-01, 9.30058975e-02, 5.80190079e-01,
            6.64651633e-01, 5.83821269e-01],
           [2.26297379e-01, 1.90022710e-01, 9.53720667e-01, 5.14270734e-01,
            4.15346281e-01, 4.96314544e-01, 4.52366417e-01, 5.94662560e-01,
            7.91297007e-01, 2.07625687e-01, 5.63166422e-01, 6.56349961e-01,
            4.71712611e-01, 2.75534331e-01, 8.49809005e-01, 1.56118581e-01,
            2.13215209e-01, 3.06627861e-01, 3.71058907e-01, 5.62302991e-01,
            3.17328855e-01, 8.73208874e-01, 4.09203959e-01, 5.71581260e-01,
            3.17169135e-01, 4.05446568e-01, 7.96942267e-01, 1.56060445e-01,
            4.42942172e-01, 8.01147077e-01, 2.59941549e-03, 8.08003680e-01,
            4.39053569e-01, 8.64538148e-01, 9.08317809e-01, 5.42054866e-01,
            4.30537684e-01, 8.00627252e-01, 7.02198618e-01, 9.07857650e-01,
            9.64071352e-01, 4.16449229e-01, 2.17517449e-01, 4.80606284e-01,
            7.12293354e-01, 1.06732510e-02, 3.10536380e-01, 9.22633818e-01,
            6.81666189e-01, 6.24167804e-01],
           [6.79900529e-01, 1.75375765e-01, 7.74831103e-01, 4.51360411e-01,
            7.84686889e-01, 4.83854048e-01, 4.53583053e-01, 8.99983658e-01,
            2.90939227e-02, 2.03570207e-01, 7.20679809e-01, 9.34854654e-01,
            8.99149364e-01, 5.63587400e-02, 5.63211048e-01, 4.87033109e-01,
            2.75330136e-01, 8.95884694e-01, 5.37172601e-01, 8.80532007e-01,
            1.41171486e-01, 8.01480329e-01, 9.00111518e-01, 8.74942736e-01,
            8.49791237e-01, 3.99779403e-01, 8.44924218e-01, 1.22134667e-01,
            6.65183025e-01, 7.30792079e-01, 9.24981066e-01, 5.96340101e-01,
            1.71169951e-01, 7.81357782e-01, 2.59912011e-01, 3.27655470e-01,
            6.74396196e-01, 4.09835467e-01, 6.23541561e-01, 4.34381794e-01,
            1.81901357e-01, 6.96795979e-01, 9.98273369e-02, 3.90759757e-01,
            4.02720018e-01, 9.43538466e-01, 9.40825161e-01, 2.56271182e-01,
            2.92165169e-01, 4.26202172e-01],
           [2.69545542e-01, 5.61694184e-01, 8.03679368e-01, 1.64893844e-01,
            3.95601910e-01, 2.91093083e-01, 5.94307198e-01, 1.22809162e-01,
            6.00669507e-01, 9.46186358e-01, 4.89280074e-02, 8.65981275e-01,
            6.11723212e-01, 2.43525055e-01, 7.93425197e-01, 8.79930422e-01,
            9.30703814e-01, 3.95002001e-01, 2.80047450e-01, 5.59755256e-01,
            7.19201792e-01, 3.20396194e-01, 8.10228635e-02, 2.77217469e-01,
            6.98872573e-01, 2.34750614e-01, 2.76818363e-01, 2.43105286e-02,
            3.98969179e-01, 5.85609177e-01, 4.99157641e-02, 3.78621719e-01,
            2.11882347e-01, 1.02583768e-01, 7.82066037e-01, 3.53437154e-01,
            7.06462578e-01, 4.41651121e-01, 9.32809328e-01, 9.50072870e-01,
            4.55079788e-01, 3.10588072e-02, 2.58934505e-01, 2.18415285e-01,
            2.81612870e-02, 7.35993055e-01, 7.92541552e-01, 9.59728281e-01,
            1.98256211e-01, 7.02334701e-01],
           [1.61601982e-01, 8.22810875e-01, 8.57382420e-01, 8.50873386e-01,
            1.75159180e-01, 9.93365268e-01, 4.58487116e-01, 8.56046915e-01,
            2.09695512e-01, 2.06523355e-01, 4.45189109e-01, 4.63978499e-01,
            6.78257832e-01, 3.60323355e-02, 1.83188961e-02, 2.13447350e-01,
            4.70807303e-01, 5.86784970e-01, 7.49137860e-01, 7.98306992e-01,
            7.44758307e-01, 9.60715199e-01, 9.09112152e-01, 2.35451493e-01,
            2.95013959e-01, 9.07803306e-01, 4.43469907e-01, 5.96816275e-01,
            9.08894465e-01, 2.26281771e-01, 7.91623177e-01, 6.31917572e-01,
            4.46217661e-01, 8.21809641e-01, 8.38562071e-01, 4.53824709e-01,
            9.92758966e-01, 7.32596228e-01, 3.26920666e-01, 7.36156045e-01,
            1.12227943e-01, 6.61665942e-01, 1.71657664e-01, 6.45049338e-01,
            8.64207521e-01, 7.88614831e-01, 9.55863205e-01, 2.17488735e-01,
            1.87450463e-01, 9.99267725e-01],
           [6.77134178e-02, 7.98943004e-01, 7.60731941e-02, 4.31534582e-02,
            5.04917078e-01, 7.91571716e-01, 6.64867343e-01, 5.60482103e-01,
            5.91621164e-01, 5.10131195e-01, 3.06537241e-01, 8.41231744e-01,
            9.74857525e-01, 8.89182445e-02, 5.00240548e-02, 2.60717570e-01,
            8.53998697e-01, 4.61732505e-01, 6.32170213e-01, 5.55705402e-01,
            3.55734775e-01, 8.03635509e-01, 7.32915671e-01, 6.46466484e-01,
            3.76342040e-01, 5.28209875e-01, 5.57619492e-01, 2.94673140e-01,
            7.00268348e-01, 4.62178177e-01, 9.39428117e-01, 7.63576114e-01,
            1.73729812e-01, 1.88256063e-01, 8.53717168e-02, 2.21594964e-01,
            1.83469270e-01, 7.26077403e-01, 3.04076658e-01, 6.95961183e-02,
            7.00311650e-01, 8.28326598e-01, 1.42210571e-01, 2.66220893e-01,
            4.71491960e-01, 1.71255982e-01, 7.60325963e-01, 7.99322264e-01,
            8.98345823e-01, 4.69914376e-03],
           [1.60392551e-01, 5.41962693e-01, 5.80370963e-01, 6.33619092e-01,
            7.96169087e-01, 4.19931270e-01, 3.49820045e-01, 8.58232840e-01,
            6.27988163e-01, 7.82169975e-01, 9.02547620e-01, 7.05029466e-02,
            4.07994048e-01, 4.92399171e-01, 1.61680470e-01, 4.79936532e-01,
            3.15943925e-01, 7.48017998e-01, 4.43633150e-02, 6.60536663e-01,
            7.78206940e-02, 5.89816935e-01, 2.06733124e-02, 5.65564014e-01,
            9.99321781e-01, 6.59645779e-01, 7.63963049e-01, 8.40377792e-01,
            6.15912592e-02, 5.93778956e-01, 8.74509137e-01, 1.82305575e-01,
            8.95003312e-01, 2.33764797e-01, 7.04413553e-01, 3.45911600e-01,
            9.77750522e-01, 6.14799922e-01, 8.07888004e-01, 5.19101916e-01,
            3.19441432e-01, 5.93872905e-01, 1.75834975e-01, 2.24650235e-01,
            2.01086076e-01, 3.80821750e-01, 7.30454687e-02, 1.41431801e-01,
            6.69907037e-01, 7.40191359e-02],
           [3.30593859e-01, 7.40289231e-01, 7.98166307e-01, 7.83321444e-01,
            8.65575729e-01, 2.98719822e-01, 1.58527585e-01, 7.08109818e-01,
            6.80408612e-01, 7.35117242e-01, 9.43761181e-01, 7.67959137e-01,
            9.97013241e-01, 3.67101011e-01, 7.63258600e-01, 7.45133816e-01,
            5.30847187e-01, 4.74846305e-01, 4.06200673e-01, 7.80749890e-01,
            6.63707802e-01, 2.60748212e-01, 8.67788019e-01, 8.54440487e-01,
            1.44999959e-02, 7.09314964e-01, 1.88188663e-01, 2.58823709e-01,
            7.95493701e-01, 8.26785449e-01, 2.77295828e-01, 2.86020608e-01,
            4.12981863e-01, 2.07092008e-02, 7.68729594e-01, 9.45194355e-01,
            7.38292744e-03, 4.23737220e-01, 6.31703107e-01, 4.46821555e-01,
            3.62170232e-01, 1.91218924e-01, 1.22643423e-01, 8.21479234e-01,
            9.02215499e-01, 1.84815071e-01, 4.01174243e-01, 3.14879922e-01,
            3.77497100e-01, 7.44965562e-01],
           [7.77053921e-01, 2.35419489e-01, 7.09578833e-01, 2.21537035e-01,
            5.01335063e-01, 7.11603835e-01, 5.94046607e-01, 4.64221570e-01,
            8.17429579e-01, 3.14442202e-01, 9.27260889e-01, 3.20730930e-01,
            2.75314918e-01, 1.85956643e-01, 2.58579123e-01, 9.22509013e-01,
            9.14877648e-01, 4.40313934e-01, 5.87450265e-01, 5.94797055e-01,
            4.13738355e-01, 1.98652500e-01, 4.41664266e-01, 2.88317727e-02,
            7.91974361e-01, 4.29381240e-01, 8.64920703e-01, 2.88044648e-01,
            6.33779159e-01, 8.84159319e-01, 5.99882141e-01, 6.53870277e-01,
            8.37018884e-01, 1.36643176e-02, 4.74381645e-01, 8.03322656e-01,
            5.39102576e-01, 2.24950397e-01, 2.52444626e-01, 3.50411071e-01,
            6.78879308e-01, 2.56271077e-01, 6.51230836e-01, 5.57034577e-02,
            3.48867799e-01, 8.33588663e-01, 6.38824005e-01, 7.99137490e-01,
            4.37760196e-01, 3.55883574e-01],
           [6.84361830e-01, 3.16652828e-01, 1.97257899e-01, 1.26678784e-01,
            7.60471767e-02, 3.99468039e-02, 6.23442327e-01, 8.99956097e-01,
            2.91457866e-01, 5.76907513e-01, 7.60338575e-01, 7.31439993e-01,
            1.15080243e-01, 8.88226949e-01, 6.87801993e-01, 4.01423651e-01,
            2.73495184e-01, 4.71115420e-01, 4.06356182e-01, 7.89409613e-01,
            1.23422717e-01, 4.66277057e-02, 5.98017748e-01, 2.95715418e-01,
            1.31222257e-01, 8.33206169e-01, 8.70153102e-01, 8.39841716e-01,
            8.90747661e-01, 6.43582815e-01, 9.26239058e-01, 2.58899402e-01,
            4.60422794e-01, 8.80037888e-01, 2.53905685e-01, 6.17429522e-01,
            3.13637282e-01, 9.77377841e-03, 3.97574982e-02, 6.21990477e-01,
            8.91024606e-01, 2.87192945e-01, 6.01163842e-01, 7.90091509e-01,
            3.62065478e-01, 9.00887675e-01, 3.51150935e-01, 7.25035058e-01,
            5.47455992e-01, 8.93993085e-02],
           [1.41617724e-01, 3.26927062e-01, 7.37450230e-01, 8.99632920e-01,
            5.17782844e-01, 5.48095240e-01, 5.76335935e-02, 1.47010595e-01,
            5.31332433e-01, 9.05799929e-01, 5.20380505e-01, 3.14046879e-01,
            5.45301668e-01, 2.79731045e-01, 3.58299592e-01, 8.79768424e-01,
            3.77795667e-01, 8.43499989e-01, 1.61886591e-01, 7.41574283e-01,
            9.99107495e-02, 9.96665026e-01, 9.86857781e-01, 3.50404680e-01,
            5.87686635e-01, 5.88893606e-01, 5.24143412e-01, 7.57274731e-01,
            6.41220915e-01, 5.35189031e-01, 9.47234170e-01, 5.60587349e-01,
            4.37504001e-01, 1.98195787e-01, 2.21247270e-01, 4.61333812e-01,
            1.63166363e-01, 1.46015916e-01, 6.87412450e-01, 2.42748813e-02,
            2.81201528e-01, 8.60424795e-01, 7.51798637e-01, 6.86093905e-01,
            3.59895278e-01, 7.71966856e-03, 8.78946826e-01, 3.96973299e-02,
            1.34644385e-01, 1.48922696e-01],
           [4.83211612e-01, 2.11342776e-01, 3.89103437e-01, 4.88028332e-01,
            6.33244344e-01, 3.19890221e-01, 8.86224029e-01, 7.81489121e-01,
            7.16106184e-01, 9.47798828e-01, 7.61932851e-01, 8.89782948e-01,
            1.52267166e-01, 6.06834899e-02, 1.32222714e-01, 6.34730241e-01,
            6.53190612e-01, 6.31511469e-02, 5.90230543e-01, 7.13097156e-01,
            2.84799306e-01, 4.08583489e-01, 3.88959627e-01, 2.49390621e-01,
            6.88725139e-02, 6.93363158e-01, 4.35914765e-01, 7.22825493e-01,
            6.12449813e-01, 8.67747596e-01, 5.80395211e-01, 3.71339637e-01,
            1.01786572e-01, 5.73240918e-01, 7.72163484e-01, 5.21016457e-01,
            3.22519750e-01, 6.29818015e-01, 6.10760201e-01, 4.92553435e-01,
            2.59542722e-01, 9.16245923e-01, 4.22026093e-01, 6.77235848e-01,
            1.91199895e-01, 2.84449660e-01, 4.34501782e-01, 2.30775641e-01,
            5.56471459e-01, 2.03636233e-01],
           [5.28042313e-01, 4.11320878e-01, 5.43472397e-01, 7.82162323e-01,
            4.01639241e-01, 8.52040787e-02, 5.81047802e-01, 4.90939472e-01,
            5.97261127e-01, 3.51904431e-01, 3.88029398e-01, 5.42341471e-01,
            8.31136123e-01, 6.98232876e-01, 4.24420813e-01, 5.55977688e-01,
            2.42794927e-01, 7.98560723e-01, 2.19193906e-02, 6.68728955e-01,
            8.92382350e-01, 2.25810722e-01, 4.32944238e-01, 9.77840962e-01,
            5.84776466e-01, 3.47453813e-01, 3.56642581e-01, 5.21073768e-01,
            2.99407001e-01, 5.68581105e-02, 1.28260341e-02, 7.82262515e-01,
            2.89734652e-02, 2.88041194e-01, 4.19198347e-01, 7.62481881e-01,
            5.92376405e-01, 7.26227339e-01, 7.86324893e-01, 2.09136571e-02,
            3.52724702e-01, 7.34158623e-01, 2.53742494e-01, 6.25225634e-01,
            5.11872721e-01, 2.27319814e-01, 1.93031813e-01, 4.44715423e-01,
            1.26344445e-01, 6.18300856e-01],
           [3.36575372e-02, 3.28284806e-01, 1.74337979e-01, 1.96717856e-01,
            6.06402568e-01, 1.99231570e-02, 9.46651758e-01, 7.90928566e-01,
            4.91134933e-01, 7.26075816e-01, 8.49137811e-02, 2.11978397e-01,
            1.95695004e-01, 3.58949207e-01, 1.46670137e-01, 4.55862629e-01,
            9.90806079e-01, 7.25271839e-01, 7.13620755e-01, 5.99640169e-01,
            3.04706807e-01, 9.58247522e-01, 1.62534947e-01, 8.72165707e-01,
            1.85745176e-01, 1.62066991e-02, 6.61045176e-01, 9.23840001e-01,
            7.17915734e-01, 4.36930158e-01, 5.61376219e-01, 6.98545993e-01,
            9.16597884e-01, 4.02984117e-01, 9.13172538e-01, 7.81888460e-01,
            3.99686245e-01, 2.08217306e-01, 7.49743174e-01, 8.88932245e-01,
            5.17586965e-01, 6.42383421e-01, 4.34892984e-01, 1.12229419e-01,
            7.10137390e-01, 9.20241124e-01, 5.25063245e-02, 2.14008777e-01,
            1.45656974e-01, 5.89479743e-01],
           [6.60335812e-01, 5.90024828e-01, 6.49940848e-01, 7.21829904e-01,
            1.37624774e-01, 8.27112220e-01, 8.90654973e-03, 6.78435553e-01,
            7.13581431e-01, 7.73961495e-01, 7.86381640e-01, 1.40685484e-01,
            6.83692065e-02, 7.40419628e-01, 9.42222499e-01, 2.29840685e-01,
            8.83614300e-01, 8.54835525e-01, 2.73841666e-01, 1.29301345e-01,
            8.97160103e-01, 4.04563706e-01, 1.07817499e-01, 9.83734260e-01,
            1.81653598e-01, 6.35045672e-01, 6.14803012e-01, 6.27673659e-01,
            6.78672764e-01, 1.03498762e-01, 4.16057356e-02, 9.49273813e-01,
            7.41503765e-01, 2.20029735e-01, 3.86060612e-01, 7.84661497e-01,
            3.69850254e-02, 9.97613212e-01, 5.87726663e-01, 2.88472957e-02,
            9.80929305e-01, 1.44050606e-01, 6.51122301e-01, 5.19613285e-01,
            7.15414651e-01, 7.48279013e-01, 3.50958714e-01, 5.16836088e-01,
            1.71377722e-01, 1.61327213e-01],
           [4.12075595e-01, 2.17900699e-01, 3.55076707e-01, 9.59830519e-01,
            5.47232274e-02, 2.23339089e-01, 5.22411055e-01, 9.65150110e-01,
            7.41732391e-01, 3.04323043e-01, 8.93653400e-01, 2.37356917e-01,
            8.38045436e-01, 5.89849595e-01, 7.46249821e-01, 1.70236936e-01,
            2.81571761e-01, 9.95551894e-02, 8.00219149e-01, 7.31865913e-01,
            9.58400083e-01, 3.80146512e-01, 3.80116585e-01, 9.62082322e-01,
            6.32716028e-01, 8.84041340e-01, 1.42174107e-01, 4.08699286e-01,
            4.82625234e-01, 3.11603529e-01, 2.10748967e-01, 5.26729960e-01,
            1.70571041e-01, 7.88818064e-01, 1.30365418e-01, 1.36499916e-01,
            1.30308219e-01, 4.26193251e-01, 1.47779766e-01, 5.57586180e-01,
            1.62465069e-01, 5.10190670e-02, 1.13472925e-01, 5.25956737e-01,
            1.97359722e-01, 3.98233236e-01, 7.52937170e-01, 9.42408671e-01,
            2.39461753e-01, 8.10087155e-01],
           [8.98743796e-02, 3.93319418e-01, 8.67469763e-01, 8.47044040e-01,
            8.47435319e-01, 1.17176908e-01, 3.36886972e-01, 8.41152881e-01,
            3.11858411e-01, 2.34784646e-01, 4.24213940e-01, 4.12674845e-01,
            3.73540059e-01, 6.62450157e-01, 3.65885620e-01, 7.41667896e-01,
            7.11859435e-02, 3.19276686e-01, 9.90213054e-01, 8.46348955e-01,
            7.07469654e-01, 5.47804787e-01, 8.45357140e-01, 9.19110201e-01,
            3.91634374e-01, 9.82410034e-01, 7.44038954e-01, 8.27533975e-01,
            3.56214080e-01, 7.76939575e-01, 4.86318543e-01, 3.26370047e-01,
            4.31871367e-01, 7.44016852e-01, 4.16789099e-01, 4.09098266e-01,
            7.46575360e-01, 5.62792261e-02, 6.57659504e-02, 7.88813467e-01,
            1.54101913e-01, 3.97015843e-01, 8.93668828e-01, 7.10012471e-01,
            3.58170535e-01, 4.53257981e-01, 1.32801695e-01, 1.53608207e-01,
            6.43725286e-01, 2.39397302e-01],
           [5.92478371e-01, 7.10386573e-02, 6.43824346e-01, 4.37153586e-01,
            5.50041803e-01, 7.32301382e-01, 7.80527238e-01, 3.12694816e-01,
            2.48292673e-01, 5.99210765e-01, 5.43997835e-01, 4.24233870e-01,
            1.66778828e-01, 5.13229559e-01, 8.07090311e-01, 2.74487079e-01,
            5.34546731e-01, 8.29036687e-01, 6.86931585e-01, 2.93729122e-01,
            7.06311706e-01, 6.57466558e-01, 6.13369677e-01, 8.61775390e-01,
            2.40748354e-01, 3.39624568e-01, 2.78622635e-01, 7.17970414e-02,
            9.90659117e-01, 2.81590797e-01, 3.66397331e-01, 2.14078646e-01,
            7.42824051e-01, 1.35995891e-01, 4.19733608e-02, 8.79760595e-01,
            1.67987788e-01, 3.54636797e-01, 3.10008705e-01, 3.62871222e-01,
            5.04966029e-01, 4.84380573e-01, 7.95457248e-02, 4.20032554e-01,
            2.13480489e-02, 5.37209984e-02, 1.32871020e-01, 1.92958663e-01,
            9.54114448e-01, 4.04372340e-01],
           [8.05879745e-01, 9.02487545e-01, 5.21292164e-01, 1.22248519e-01,
            3.93834242e-01, 8.97451946e-01, 1.26920820e-02, 9.63549754e-01,
            6.48947841e-01, 5.82222807e-01, 6.46400557e-01, 7.78306066e-01,
            6.11904887e-01, 1.34928446e-01, 9.14806380e-01, 3.11983886e-01,
            9.68564057e-01, 2.24152252e-01, 8.30691840e-01, 3.68626664e-01,
            9.20400704e-01, 1.89308914e-01, 4.32956656e-01, 9.90321701e-01,
            3.58273162e-01, 7.10955101e-01, 2.69503379e-01, 9.37914984e-01,
            5.93787966e-01, 4.30482244e-01, 9.66779309e-01, 4.62536889e-01,
            8.28494133e-01, 3.60502875e-01, 9.30781040e-01, 5.27588299e-01,
            3.81180689e-01, 7.71739635e-01, 3.71208160e-01, 1.55923174e-01,
            4.90518222e-01, 7.80761612e-01, 4.84916383e-01, 8.72735794e-01,
            2.35262208e-01, 2.39182303e-02, 1.70430238e-01, 2.58213049e-01,
            7.42570981e-01, 8.57315083e-01],
           [3.78637004e-01, 7.06029925e-01, 3.67714518e-02, 2.91109303e-01,
            8.25940750e-01, 2.96454524e-01, 7.40251167e-01, 9.75473325e-01,
            6.01355530e-01, 9.59431582e-01, 5.08018680e-01, 7.45247439e-01,
            1.48648806e-01, 5.26337365e-02, 8.62423675e-01, 9.12985188e-01,
            4.16586737e-01, 1.55633370e-01, 8.12244714e-01, 4.94405258e-01,
            5.88303573e-01, 5.84922648e-01, 5.73994204e-01, 3.16627542e-01,
            1.64149496e-02, 4.77506196e-01, 9.53456486e-01, 1.83976006e-01,
            5.99057154e-01, 1.27598759e-01, 8.54932136e-01, 1.33327996e-01,
            7.25956077e-01, 2.67633363e-01, 1.85201179e-01, 1.52416410e-01,
            5.23768657e-01, 5.18796429e-01, 2.51048248e-01, 7.45534081e-01,
            7.79121647e-01, 5.69040139e-01, 7.68762413e-01, 6.21715699e-01,
            4.67321421e-01, 7.82091490e-01, 9.18734500e-01, 3.21732444e-01,
            7.70242133e-01, 5.61629744e-01],
           [3.26231074e-01, 2.86861910e-01, 5.68731038e-01, 5.23030448e-01,
            6.93229991e-01, 1.86302847e-01, 1.02598615e-01, 7.82906691e-01,
            2.25924779e-01, 6.95240972e-01, 2.28906149e-01, 1.71802263e-01,
            2.00421378e-01, 7.05552421e-01, 8.67238656e-01, 5.74599352e-01,
            2.88871884e-01, 4.75244802e-01, 5.65055068e-01, 4.72919972e-01,
            5.68022017e-01, 3.10222669e-01, 6.07368176e-01, 9.97920264e-01,
            2.12876986e-02, 7.96269926e-01, 3.06637682e-01, 5.44683767e-01,
            8.13148721e-01, 6.02823428e-01, 9.83427058e-01, 9.29673840e-01,
            1.95066438e-01, 6.20013585e-01, 4.22968015e-01, 6.79969793e-01,
            2.45124465e-01, 8.35467967e-01, 5.04620537e-01, 5.48662464e-01,
            1.29728716e-01, 8.07737242e-01, 3.07115715e-01, 2.78329013e-01,
            5.23942105e-01, 1.29304757e-01, 1.86638057e-01, 8.69679338e-01,
            7.98436451e-01, 6.55591732e-01],
           [4.82511809e-01, 2.66054600e-01, 2.42780615e-02, 3.53509827e-01,
            7.97089900e-03, 2.36605642e-01, 2.65821101e-01, 5.97227352e-01,
            7.84152372e-01, 9.17176006e-01, 6.96059567e-01, 5.61981345e-01,
            2.77069620e-01, 4.58277801e-01, 5.42093325e-02, 9.11540275e-01,
            9.82658680e-02, 2.58616683e-01, 1.15678338e-01, 6.44295110e-01,
            1.41292658e-01, 9.55339681e-01, 9.18835649e-01, 1.80581786e-01,
            6.54665249e-01, 4.21558516e-01, 8.27160856e-01, 6.16151805e-01,
            1.80496016e-01, 7.96135434e-01, 7.92771500e-01, 8.39724201e-01,
            2.26053672e-01, 7.18529337e-01, 8.01843393e-01, 3.27201526e-01,
            3.42405196e-03, 2.64285076e-01, 4.18077968e-01, 5.58529030e-01,
            1.20624091e-01, 1.88947971e-01, 9.51877899e-01, 9.14027997e-01,
            2.60244873e-01, 5.87403742e-01, 6.73820421e-01, 7.67253443e-01,
            2.01166371e-01, 1.34691702e-01],
           [6.49530972e-01, 5.02199364e-02, 2.53575261e-01, 2.74647194e-01,
            1.58952376e-01, 7.24724109e-01, 8.95431085e-01, 6.87151658e-01,
            8.63698938e-01, 6.78888018e-01, 8.64306578e-01, 4.14001212e-02,
            7.71836190e-01, 5.06367961e-01, 5.08737247e-01, 9.04501193e-01,
            1.76456317e-01, 1.18610668e-02, 1.63570143e-01, 1.90361250e-01,
            1.23885101e-01, 3.48241664e-01, 3.77872706e-01, 6.52316468e-01,
            6.32991823e-01, 3.51716520e-01, 7.15706253e-01, 9.89268377e-01,
            6.62415154e-01, 5.73149141e-01, 7.70069694e-01, 5.44650782e-01,
            5.13901352e-01, 7.70927542e-01, 8.19342672e-01, 9.22794671e-01,
            5.69373660e-02, 2.27219361e-01, 3.79948534e-01, 6.82911579e-01,
            1.21980135e-01, 5.78915187e-01, 9.97900298e-02, 3.23915824e-02,
            5.50844764e-03, 8.72689839e-01, 3.29136229e-01, 7.92236161e-02,
            9.38884842e-01, 7.38260139e-01],
           [4.94360428e-01, 1.03558746e-01, 3.17316527e-02, 3.05614713e-01,
            2.66055152e-01, 7.05146366e-01, 4.55481800e-01, 9.66495342e-01,
            7.93175837e-01, 7.37773406e-02, 2.72047360e-01, 7.87452036e-01,
            5.21251506e-01, 2.94650040e-01, 2.82343610e-01, 8.13731391e-01,
            5.53945065e-01, 4.23775249e-01, 3.51794516e-01, 7.20660209e-01,
            7.74859009e-02, 9.31206333e-01, 3.97680817e-01, 6.51535472e-01,
            4.19334384e-01, 3.65948638e-01, 7.31140806e-01, 3.43629156e-01,
            9.00930793e-02, 5.80093651e-02, 5.70139457e-01, 9.02487063e-01,
            8.90036056e-03, 1.17754995e-01, 2.98489693e-01, 3.14390103e-02,
            7.94638114e-01, 7.30042223e-01, 3.70064123e-01, 7.86345453e-01,
            4.85150573e-01, 7.18064393e-01, 2.17741755e-01, 1.40080478e-03,
            9.31104308e-01, 4.98848708e-01, 4.78951322e-02, 6.33677802e-01,
            3.50582735e-01, 7.58812339e-01],
           [6.64298180e-01, 4.48134136e-01, 5.40221728e-02, 4.15713827e-03,
            4.73529144e-02, 3.24834272e-01, 1.48990508e-01, 5.98453323e-01,
            3.51049310e-01, 1.47205752e-01, 5.87700912e-01, 7.29046135e-01,
            4.68816832e-01, 8.18945463e-02, 1.26726252e-01, 5.74167470e-01,
            9.40781132e-01, 4.49903491e-01, 2.24380925e-01, 3.10749826e-01,
            8.04094268e-01, 3.09511112e-01, 4.57434275e-01, 3.75617563e-01,
            5.71064562e-02, 5.50079558e-01, 4.44212104e-01, 7.77155064e-01,
            2.72738179e-01, 3.50739275e-01, 6.50776902e-01, 8.87102060e-01,
            3.93688738e-01, 7.46466066e-01, 9.27755859e-01, 5.22213051e-01,
            3.71458513e-01, 5.87165920e-01, 6.58633973e-01, 8.65281658e-01,
            8.89576662e-01, 3.23110054e-01, 3.64237399e-01, 2.69940448e-01,
            5.79820990e-01, 4.75327762e-01, 8.94432652e-01, 7.78382183e-02,
            6.94581555e-01, 4.83675577e-01],
           [5.32973180e-01, 8.92026344e-01, 5.05390928e-02, 1.68257752e-01,
            8.93138785e-01, 9.21862785e-01, 9.59631844e-01, 7.80442489e-01,
            4.75536322e-01, 1.28115503e-01, 9.77614409e-01, 8.07821346e-01,
            1.79455263e-01, 2.49231464e-01, 7.30633421e-01, 6.49643453e-01,
            5.28831188e-02, 6.56147442e-01, 7.74642467e-01, 6.25918669e-01,
            1.36226100e-01, 6.02651503e-01, 5.88287596e-01, 7.14673404e-01,
            1.87829972e-01, 7.53472814e-01, 3.12259089e-01, 7.66906837e-01,
            9.24047973e-01, 1.12139627e-01, 7.34354219e-01, 8.02892261e-01,
            9.63782968e-02, 2.71216198e-01, 2.89517519e-01, 8.52599127e-01,
            1.62309792e-01, 2.88365281e-01, 7.86824469e-01, 6.05109914e-01,
            4.58678077e-01, 6.42330101e-01, 5.47335284e-01, 3.14345921e-01,
            1.65936483e-01, 3.01871238e-01, 8.44254222e-01, 6.41537396e-01,
            1.66742060e-01, 7.54753939e-01],
           [8.19768572e-01, 1.24943677e-01, 2.33175491e-01, 1.91034915e-02,
            1.33178916e-01, 2.52742429e-01, 5.12430412e-01, 3.10073922e-01,
            8.36689379e-01, 6.15859211e-01, 4.79470686e-01, 2.87390564e-01,
            4.55271037e-02, 9.21188119e-01, 3.55621070e-01, 8.81364052e-01,
            5.19543280e-01, 2.91936517e-01, 6.31556934e-01, 4.22156580e-01,
            8.33871994e-01, 1.87318396e-01, 7.48329444e-01, 4.16517801e-03,
            4.34322368e-02, 9.73960201e-01, 6.08745251e-01, 1.62335127e-01,
            9.09819173e-02, 1.03452739e-01, 8.52554333e-01, 7.94041491e-02,
            2.44723333e-01, 1.94306582e-01, 7.09629356e-01, 5.67432247e-01,
            3.99027958e-01, 3.62690709e-01, 7.93570915e-01, 2.28403051e-01,
            4.42390752e-01, 3.52210599e-01, 9.57329078e-01, 1.36428313e-01,
            2.86334427e-02, 5.79196322e-01, 6.16004787e-01, 1.79844125e-01,
            7.97899272e-02, 8.89127390e-01],
           [9.48870138e-01, 1.76315058e-01, 1.46754297e-01, 7.36010899e-01,
            6.21041269e-01, 1.07500960e-01, 2.20622360e-01, 6.07546784e-01,
            7.35378754e-01, 7.80417336e-01, 5.00510630e-01, 7.99200179e-01,
            5.27987010e-01, 9.62864967e-01, 5.43792860e-01, 8.56957927e-01,
            9.82819439e-01, 1.83754925e-01, 3.72543131e-01, 4.56799610e-01,
            2.41910631e-01, 4.75922252e-02, 8.83353524e-01, 6.42575544e-01,
            4.49106621e-01, 6.17295538e-01, 6.03524935e-01, 8.13733762e-01,
            2.47428830e-01, 9.54121051e-01, 7.07907705e-01, 6.82999229e-01,
            3.68114641e-01, 2.57441549e-01, 7.04024133e-01, 9.00871569e-01,
            1.67427942e-01, 6.78761324e-01, 2.09743074e-01, 5.81888362e-01,
            5.80884146e-01, 4.07756745e-01, 8.93696322e-01, 8.14044484e-01,
            1.06057808e-01, 5.64546123e-01, 9.06198657e-01, 3.91225943e-01,
            8.01672827e-01, 2.38335075e-01],
           [8.59950980e-01, 1.85091534e-01, 2.82288353e-01, 2.46907229e-01,
            7.27067686e-01, 2.11703243e-02, 1.10021325e-03, 5.84172702e-01,
            8.75798660e-01, 1.06481615e-01, 9.14070810e-02, 2.09663148e-01,
            9.48718962e-01, 8.13095052e-01, 7.26726469e-01, 8.97068581e-01,
            3.83157624e-01, 4.81488043e-01, 7.68767675e-01, 9.91773247e-01,
            2.35335829e-01, 8.91896723e-01, 6.16467244e-01, 4.09699981e-01,
            3.82360927e-01, 4.48453799e-01, 2.10999381e-01, 8.86483065e-01,
            7.67044279e-01, 7.31271321e-01, 3.16819439e-01, 1.82101537e-01,
            7.14779065e-01, 4.28881562e-01, 5.11690483e-01, 8.00094793e-03,
            6.78340634e-01, 5.78531504e-01, 3.14011038e-01, 4.41427767e-02,
            5.66721437e-01, 9.53193650e-01, 7.32164846e-01, 6.49544208e-02,
            8.50542953e-01, 9.62424273e-01, 5.02664136e-01, 5.51869993e-01,
            7.39756995e-01, 6.73593132e-01],
           [2.21468371e-01, 6.82529696e-01, 1.54419941e-01, 7.47373026e-01,
            4.26441027e-01, 2.25843555e-01, 5.19570558e-01, 9.77858836e-01,
            2.75422599e-01, 2.47171738e-01, 8.81533096e-01, 4.41846451e-01,
            2.38266420e-01, 6.64592957e-01, 5.86150176e-01, 8.34738962e-01,
            6.86837733e-01, 8.30071568e-01, 7.54083032e-01, 9.47480128e-02,
            3.14677012e-01, 4.12037967e-02, 4.19330611e-01, 1.05115647e-01,
            5.10466490e-01, 9.56043361e-02, 6.03028321e-01, 7.09911106e-02,
            6.99281277e-01, 8.10313539e-01, 9.60674257e-01, 6.39095295e-01,
            9.40594104e-01, 4.01508589e-01, 9.80497538e-01, 4.46246222e-01,
            4.16778301e-01, 4.76258657e-01, 4.01213191e-02, 2.00378857e-01,
            3.14792794e-01, 3.95707525e-01, 2.76452600e-01, 5.06984216e-01,
            5.41419890e-01, 2.25459226e-01, 7.14342214e-01, 4.65397314e-01,
            4.54868978e-01, 4.83640337e-01],
           [2.11723113e-01, 1.18020937e-01, 5.74394843e-01, 8.69534374e-01,
            4.82339342e-01, 8.43486876e-01, 9.09842802e-01, 5.32612353e-01,
            2.56363340e-01, 7.92744118e-01, 9.01600159e-01, 1.80116772e-01,
            5.35315472e-01, 2.67447427e-01, 2.46667470e-01, 6.69541409e-01,
            6.51101187e-02, 4.90682956e-01, 7.15824671e-01, 3.58057461e-01,
            5.55041258e-01, 8.81707621e-01, 4.77350057e-02, 8.26420660e-01,
            6.24315104e-01, 3.57662328e-01, 8.78390991e-01, 5.24480601e-01,
            6.83730700e-01, 1.14068528e-01, 4.40325932e-01, 5.87051129e-01,
            6.52122481e-01, 8.63816990e-01, 3.22548122e-01, 3.97400424e-01,
            6.03810585e-01, 1.18216101e-01, 2.13552987e-01, 7.50158531e-01,
            5.22924822e-01, 4.36510532e-02, 8.32066793e-01, 4.45261122e-01,
            7.06814188e-01, 8.61820929e-02, 7.82866668e-01, 8.67835431e-02,
            6.83776476e-01, 8.54748189e-01],
           [7.28414023e-01, 6.28000219e-01, 6.07992183e-01, 7.95909939e-01,
            2.66618911e-01, 9.79368309e-01, 4.91486078e-02, 1.79660821e-01,
            2.59526006e-01, 1.18172380e-02, 8.56117776e-01, 5.69047533e-01,
            2.26618149e-01, 4.59743669e-01, 2.49983211e-01, 1.98335469e-01,
            5.02006590e-01, 2.32434506e-01, 7.52372617e-01, 9.31146136e-01,
            3.44611996e-01, 3.16541541e-01, 6.27970487e-01, 2.05524782e-01,
            9.09938556e-01, 1.34475621e-01, 9.57497818e-01, 6.08814433e-02,
            8.21203225e-01, 6.77704908e-01, 2.11604285e-01, 7.47502293e-01,
            5.36488317e-01, 6.68349651e-01, 4.15808559e-01, 4.50734776e-01,
            6.15595703e-01, 9.19762327e-01, 3.83088277e-01, 6.39582719e-01,
            4.94882339e-01, 4.09994549e-01, 3.37461967e-01, 7.05128212e-01,
            8.13299193e-01, 2.52582290e-01, 6.34228562e-01, 4.21050816e-01,
            6.01985950e-01, 4.25845504e-01],
           [3.00968372e-03, 2.54182752e-01, 2.67143151e-01, 6.27414651e-01,
            9.63603434e-01, 7.15642862e-01, 2.40487949e-02, 9.44879763e-01,
            3.98285234e-01, 4.67397788e-01, 4.78829296e-01, 4.32287674e-01,
            6.84937482e-01, 2.34680041e-01, 1.48690614e-01, 4.13198413e-01,
            3.43646283e-01, 2.29472247e-01, 8.55011154e-01, 6.18337534e-01,
            6.94396050e-01, 6.35106826e-01, 1.19580524e-01, 7.92073704e-01,
            5.51422947e-02, 4.66614657e-02, 7.26222624e-01, 1.45107096e-01,
            3.18494351e-01, 5.58340712e-02, 2.70614024e-02, 8.63075270e-01,
            4.22202946e-02, 5.28638108e-01, 6.69593813e-01, 8.35029062e-01,
            2.55977306e-01, 4.74062826e-01, 6.41329127e-01, 8.02311107e-01,
            4.90889419e-01, 1.36601777e-01, 9.38832921e-01, 1.32769206e-01,
            6.17844823e-01, 4.31022954e-01, 5.09317906e-01, 2.74034724e-01,
            5.52861026e-01, 1.04424839e-01],
           [5.09130840e-01, 1.11191732e-01, 5.69523846e-01, 2.98006424e-01,
            7.53137318e-01, 7.74417545e-01, 8.08076428e-01, 1.25636263e-01,
            4.17184228e-01, 5.26947314e-01, 2.21750371e-01, 6.92734094e-01,
            8.01542642e-01, 2.54439792e-01, 9.09707435e-01, 1.46527661e-01,
            2.41860567e-01, 5.61130169e-01, 9.89882381e-01, 4.17289507e-01,
            5.43849654e-01, 5.19202113e-01, 2.00671152e-01, 2.31641014e-01,
            9.99252808e-01, 4.90781148e-01, 1.28612614e-01, 3.75219870e-01,
            1.23694998e-01, 4.72090334e-02, 8.35462837e-01, 1.84789770e-01,
            5.56268233e-01, 5.85902535e-01, 2.72172863e-01, 4.80343575e-01,
            7.09086825e-01, 8.44029814e-01, 2.06129591e-01, 8.00175425e-01,
            5.30047394e-01, 3.04658191e-01, 6.34426412e-01, 5.47884901e-01,
            3.16001428e-01, 5.07308928e-01, 4.82682189e-01, 3.33050513e-02,
            8.79484407e-01, 4.93907887e-01],
           [7.22887546e-01, 4.60424458e-02, 5.50645174e-02, 5.72517616e-01,
            6.30597302e-01, 8.45909056e-01, 4.49202020e-01, 2.00334975e-01,
            2.05563953e-01, 7.11116964e-01, 3.89918685e-02, 4.51213943e-01,
            5.93794167e-01, 1.96395169e-01, 7.75917144e-01, 8.53433054e-01,
            6.79200427e-01, 2.49601221e-01, 5.17664097e-01, 1.26392566e-01,
            1.90932852e-01, 8.26646132e-01, 4.16072368e-01, 1.91565390e-01,
            7.77555613e-02, 5.91286329e-02, 5.45122690e-01, 8.31120273e-01,
            9.03241874e-01, 6.28707407e-01, 9.68175002e-01, 3.46013116e-01,
            4.08853292e-01, 7.61106795e-01, 1.38843107e-01, 5.38939794e-01,
            9.22128641e-01, 7.04794363e-01, 7.22885594e-01, 8.19896368e-01,
            8.06882195e-01, 8.88721656e-01, 8.64242045e-01, 8.83438565e-01,
            3.21815489e-01, 6.37086315e-01, 6.80500611e-01, 3.33667398e-02,
            2.51296245e-01, 3.97747281e-02],
           [6.80582558e-02, 3.48867917e-01, 5.12070701e-01, 1.37063577e-01,
            9.03886141e-01, 8.24824456e-02, 7.52606365e-01, 9.14707682e-01,
            1.11413113e-01, 1.85768170e-01, 1.46663118e-01, 9.19234950e-01,
            2.80589136e-02, 1.77516043e-01, 9.11692336e-02, 4.02002145e-01,
            1.59929375e-01, 2.81453083e-01, 6.19749877e-01, 4.79425852e-01,
            8.95315378e-01, 7.92429361e-01, 9.76826424e-01, 9.76427853e-01,
            5.33708622e-01, 8.23082046e-01, 9.02361247e-01, 1.91448089e-01,
            1.20352973e-01, 6.99367457e-01, 8.75767733e-02, 9.31573904e-01,
            6.66273641e-01, 1.54517801e-01, 5.06892853e-01, 5.30220565e-01,
            7.57355432e-01, 1.96201957e-01, 2.07069685e-01, 2.10275671e-01,
            4.46928701e-01, 5.50209110e-01, 3.14404087e-01, 6.87559058e-01,
            6.54896677e-01, 8.50444679e-01, 9.61750063e-01, 6.50639936e-01,
            4.01911200e-01, 3.26131535e-01],
           [6.88868682e-01, 7.35774371e-01, 4.34528404e-01, 8.22063175e-01,
            2.08471452e-01, 1.01168246e-01, 1.46223616e-01, 7.17441057e-01,
            4.64033829e-01, 1.87296476e-01, 9.44689854e-01, 6.40719311e-01,
            1.47128662e-01, 8.61825267e-01, 7.95629160e-01, 7.10037631e-01,
            1.31723561e-01, 3.01955364e-02, 1.18713866e-01, 5.75527383e-02,
            5.75515571e-01, 8.91998517e-01, 5.40889715e-02, 8.56228985e-01,
            6.79087707e-01, 9.57043632e-01, 3.53060098e-01, 3.18918693e-01,
            8.48809784e-02, 2.18214165e-01, 1.98060079e-01, 1.04404818e-01,
            9.24642053e-01, 6.86439816e-01, 7.65896346e-02, 5.74776833e-02,
            3.46117595e-01, 3.52735142e-01, 5.70174849e-01, 8.42072865e-01,
            9.90374619e-01, 2.85972629e-01, 9.82916800e-01, 2.29720739e-01,
            4.96093379e-01, 2.15388413e-01, 2.09083317e-01, 1.48193572e-02,
            7.99991627e-01, 6.18368062e-01],
           [2.14332756e-01, 1.46565592e-01, 6.74120677e-01, 6.75158962e-01,
            3.72233094e-01, 9.70898072e-01, 8.81900805e-01, 2.60248982e-01,
            1.07689016e-02, 4.19663871e-01, 4.15055546e-01, 6.74237442e-01,
            4.26250710e-01, 7.70863234e-01, 1.48543574e-01, 7.97619993e-01,
            5.61641266e-01, 2.25665982e-01, 2.00670835e-02, 5.78143701e-01,
            3.88555609e-01, 2.67805971e-01, 4.64043162e-03, 8.77480891e-01,
            3.85251983e-01, 4.19603975e-01, 5.96399156e-01, 2.03756679e-01,
            8.74071797e-01, 2.44856006e-01, 3.89017651e-01, 6.41016669e-01,
            3.86509118e-01, 3.80314033e-01, 6.50013243e-01, 1.09622345e-01,
            9.68253118e-02, 5.96671841e-01, 8.91136198e-01, 7.19129201e-01,
            9.82442482e-01, 2.68437362e-01, 3.88988530e-01, 3.80945618e-01,
            3.24215853e-02, 7.01045416e-01, 8.07348572e-01, 3.09256355e-01,
            1.10330628e-01, 2.37757246e-02],
           [4.07314995e-01, 6.23814313e-01, 3.10657855e-01, 9.69369228e-01,
            8.54290382e-01, 2.91514683e-01, 3.26246266e-01, 2.66657538e-01,
            4.60254734e-01, 1.66903378e-01, 7.86095320e-02, 9.96580655e-01,
            5.80314630e-01, 8.10559982e-01, 2.09320607e-01, 1.06288219e-01,
            8.04168670e-01, 1.26619728e-01, 6.88982080e-01, 6.03428951e-01,
            3.37469298e-01, 6.83154773e-01, 1.24974366e-03, 1.89481603e-01,
            9.72252074e-01, 1.62812432e-01, 4.60942932e-01, 9.31947987e-01,
            8.59584029e-03, 3.14000566e-01, 3.92436545e-01, 5.78641408e-01,
            8.94039839e-01, 3.22495814e-01, 7.91799336e-01, 7.56238398e-01,
            8.05300495e-01, 7.30900820e-01, 4.37874156e-01, 8.92822439e-01,
            7.23897177e-01, 6.42862955e-01, 1.59368914e-01, 8.53179250e-01,
            5.12702228e-01, 8.65738286e-01, 1.63775543e-01, 4.69664757e-01,
            4.42050686e-01, 2.40456324e-01],
           [1.89232974e-01, 8.61294438e-01, 9.23431526e-01, 3.46840539e-01,
            9.24594506e-01, 7.26856721e-01, 2.39028940e-01, 3.39937237e-01,
            3.58232128e-01, 7.39567065e-01, 3.75268339e-01, 5.65542213e-01,
            5.21416414e-01, 4.01162074e-01, 6.22820659e-01, 3.12993908e-01,
            2.87706538e-01, 4.69333861e-01, 8.98453202e-01, 1.77696589e-01,
            5.71609923e-01, 1.09655719e-01, 9.38571569e-01, 9.65652703e-01,
            8.94713113e-01, 2.01789908e-01, 8.73009889e-01, 7.62406592e-01,
            6.65265894e-01, 3.71892007e-01, 2.28206454e-02, 4.99183914e-01,
            6.08103989e-01, 1.72847387e-01, 2.44742045e-01, 3.94994023e-01,
            1.54988714e-01, 7.00379325e-01, 5.75788236e-01, 9.83863750e-02,
            8.14796283e-01, 5.87117610e-02, 3.44782078e-01, 4.12823262e-01,
            4.33315577e-01, 2.41017871e-01, 9.17566898e-01, 5.31643031e-01,
            8.50996943e-01, 4.43364885e-01],
           [9.90482472e-01, 7.78421569e-01, 9.81286344e-01, 6.53727808e-01,
            3.68864184e-01, 3.13389030e-01, 2.90938577e-01, 6.65767593e-01,
            5.18884145e-01, 4.35739663e-01, 5.44034915e-01, 3.37086711e-02,
            7.92174804e-01, 8.96716175e-01, 7.57851729e-01, 8.52732599e-01,
            8.82971351e-02, 7.55069403e-01, 3.63760028e-01, 8.22339351e-01,
            1.16406692e-02, 8.47633966e-01, 2.64121574e-01, 8.33342220e-01,
            4.60009716e-01, 1.84553485e-01, 8.55387723e-01, 8.17118238e-01,
            4.17935251e-01, 8.10935412e-01, 5.56807746e-01, 2.52673347e-01,
            6.70030474e-01, 5.95122148e-02, 4.73671851e-01, 4.33803282e-01,
            1.41294002e-01, 5.94169965e-02, 7.15228743e-01, 9.53522856e-02,
            8.90669092e-01, 9.12165426e-01, 5.70257204e-01, 2.99217474e-01,
            3.47128470e-01, 9.75168128e-01, 1.34625923e-01, 9.42069519e-01,
            9.76001148e-01, 1.51254238e-01],
           [4.03168054e-01, 3.95690991e-01, 4.90015916e-02, 1.77316880e-01,
            7.39750926e-01, 4.91618402e-01, 7.16056938e-01, 8.48815021e-01,
            5.99150158e-02, 1.52799205e-01, 2.16515417e-01, 6.28681628e-01,
            3.16947148e-01, 9.19609929e-02, 6.82589463e-01, 3.88757498e-01,
            4.23497697e-01, 9.36054219e-01, 2.63130301e-01, 4.61320950e-01,
            8.70614209e-01, 8.15257059e-01, 7.35991228e-01, 7.27697175e-01,
            9.04660373e-01, 9.83861843e-01, 4.99398750e-01, 4.46526811e-01,
            2.30547110e-01, 6.13690243e-01, 7.98055470e-01, 6.51295593e-01,
            2.94839448e-01, 1.09887805e-02, 2.80827236e-01, 4.65303323e-01,
            3.53858425e-01, 8.83293197e-01, 1.13647193e-01, 2.53900762e-01,
            4.06208541e-01, 5.22090706e-01, 3.71352528e-01, 9.78161366e-01,
            5.43947105e-01, 7.41213168e-01, 8.36391655e-02, 8.65328843e-01,
            9.43465721e-01, 1.35656045e-01],
           [7.79004061e-01, 6.97893617e-01, 5.66305257e-01, 6.27112500e-01,
            2.94437708e-01, 9.33351256e-01, 3.83361437e-01, 8.39088308e-01,
            9.05291708e-01, 4.37490654e-01, 7.23489939e-01, 5.04964350e-01,
            8.92014614e-01, 6.56463506e-01, 6.58420491e-02, 7.14017913e-01,
            4.25072788e-01, 1.00761092e-01, 5.47661996e-01, 6.81529843e-01,
            5.44666616e-01, 4.77917478e-01, 1.96379283e-01, 1.36731134e-01,
            2.91615629e-01, 2.70237439e-01, 9.34585008e-01, 4.33885969e-01,
            5.14261936e-01, 7.75462862e-01, 7.08908495e-01, 5.79824247e-01,
            2.30575071e-01, 4.86472608e-01, 4.28397299e-01, 6.55590007e-01,
            4.60494966e-01, 2.22568150e-01, 3.60020503e-01, 9.09277739e-02,
            9.20812106e-01, 9.84244022e-03, 4.46825420e-02, 6.17487261e-01,
            8.09600854e-02, 8.00100980e-01, 2.88511312e-01, 7.41628329e-01,
            4.33634865e-01, 5.35114849e-01],
           [6.26121371e-01, 4.91955020e-01, 5.87384693e-01, 3.56949008e-02,
            4.10578162e-01, 6.68803989e-01, 9.47424878e-01, 3.46864689e-01,
            3.89979756e-01, 5.54772920e-01, 9.14836293e-01, 9.24236026e-03,
            4.51421925e-01, 5.53389329e-01, 8.35928398e-02, 1.87181451e-01,
            9.64022648e-01, 8.12808893e-01, 7.25301861e-01, 6.57469017e-01,
            2.55638990e-01, 5.14077141e-01, 7.57308725e-01, 4.36858356e-01,
            5.75110579e-01, 5.56969553e-01, 2.60245449e-01, 5.37752654e-01,
            9.55558246e-01, 6.19755793e-01, 5.25941873e-01, 5.96109822e-01,
            5.72884008e-01, 6.56443197e-03, 2.14980056e-01, 1.86122619e-01,
            7.53267313e-01, 7.19209356e-01, 9.72341144e-01, 4.48036759e-01,
            5.07992571e-01, 1.66642089e-02, 5.45422435e-01, 7.42916772e-01,
            1.42832266e-01, 5.41663915e-01, 3.70217330e-01, 1.23395860e-01,
            7.14744522e-01, 5.32380707e-01],
           [8.53060845e-01, 8.89868357e-01, 4.92233622e-01, 9.21376624e-02,
            3.32904750e-01, 5.99252495e-01, 8.35415400e-01, 1.01449014e-01,
            8.45199949e-02, 8.28393410e-01, 9.60338124e-01, 8.70991096e-01,
            9.49444725e-01, 2.57270394e-01, 5.96272335e-01, 8.67143703e-01,
            6.93525858e-01, 3.89976918e-01, 1.97752276e-01, 3.51479336e-03,
            1.98850959e-01, 7.15290602e-01, 9.46090263e-01, 6.72586527e-01,
            6.74767016e-01, 9.35034671e-01, 3.81249065e-01, 5.70280153e-01,
            1.10508919e-01, 7.55895046e-01, 5.81063332e-01, 4.87136330e-01,
            7.20992908e-01, 2.97851774e-01, 2.71516470e-01, 9.00410626e-01,
            5.22963654e-01, 1.57930660e-02, 6.26126273e-01, 2.33563869e-01,
            7.99212842e-02, 3.31294846e-02, 6.73446174e-01, 8.56326738e-01,
            7.17222399e-01, 9.82680061e-01, 6.99817880e-01, 5.80988149e-01,
            8.65116696e-01, 6.01000908e-01]])




```python
# https://www.tutorialspoint.com/print-full-numpy-array-without-truncation
print(np.array2string(x, threshold = np.inf))
```

    [[9.09493927e-01 2.39865472e-02 2.55512699e-01 5.86328195e-01
      7.11310838e-02 3.43906434e-01 1.55961957e-02 4.27471885e-01
      7.55280892e-01 8.91569347e-01 6.85674566e-01 3.53757164e-01
      1.10756416e-01 2.19419896e-01 1.48348921e-01 8.86062128e-01
      3.29159187e-01 3.42868380e-02 7.20196429e-01 9.40518704e-01
      1.91526779e-01 5.78030490e-01 9.09834624e-01 3.31523512e-01
      6.25675801e-01 8.29793218e-01 1.98280128e-01 4.94173211e-01
      8.84756044e-01 5.16241643e-01 4.12229321e-01 2.69868841e-02
      2.62166565e-01 8.43344805e-01 5.28991601e-01 3.51315584e-01
      6.04624653e-01 6.07264748e-01 9.48114311e-01 4.78166438e-01
      6.68048594e-01 8.59508647e-01 2.65764461e-01 9.68481541e-01
      1.93221498e-01 4.45372291e-01 9.15174093e-01 2.46202160e-01
      5.89698895e-01 3.21986029e-02]
     [8.77345607e-01 6.58711207e-01 9.02516252e-01 9.87827993e-01
      5.40404639e-01 4.68443738e-01 1.78858010e-01 8.66190209e-01
      5.84171129e-01 7.28585681e-01 5.76629159e-01 6.28880944e-01
      8.99590495e-01 2.58480230e-01 4.28828180e-01 3.25574100e-01
      3.88793491e-01 9.47620604e-01 5.96951521e-01 4.44672279e-01
      5.51454680e-01 1.15206288e-01 1.42543001e-01 6.21700899e-01
      9.79686413e-01 2.67082399e-02 3.32875674e-01 7.29076235e-01
      4.06204298e-04 4.80769468e-03 7.68993273e-01 3.02584403e-01
      1.48593816e-01 9.40445524e-01 1.86897046e-01 1.21953422e-01
      6.03991981e-01 5.03449853e-01 2.76685087e-01 3.65547826e-01
      4.24615207e-01 7.63027836e-01 6.68590876e-01 3.47521237e-01
      7.83400712e-01 9.72431596e-02 7.75169006e-01 1.08304787e-01
      1.70414332e-01 9.35440419e-01]
     [9.00761296e-01 4.24074718e-01 4.57967226e-02 6.69926822e-01
      8.13062571e-02 4.24992824e-01 1.57816435e-01 4.49856267e-01
      8.67017093e-01 5.85894195e-01 5.58185041e-01 3.79443337e-01
      3.06991103e-01 9.13087437e-01 8.53450577e-01 1.14077617e-01
      2.69261506e-01 2.44527850e-01 5.62002921e-01 3.21048162e-01
      7.03234477e-01 6.96583530e-01 7.00936075e-01 2.86291409e-01
      3.47557687e-01 7.36594848e-01 1.53182815e-01 4.08231157e-01
      4.05792766e-02 6.13423617e-01 2.29961660e-01 9.56343702e-01
      8.88213692e-01 3.07395235e-01 9.69745029e-01 8.09560031e-01
      8.68280755e-01 7.60896385e-01 4.66355383e-01 5.35764485e-01
      4.06991140e-01 2.40143821e-01 7.49911318e-01 6.10924384e-01
      5.97709273e-01 7.66935214e-01 1.68155516e-02 5.32416390e-01
      3.97378127e-01 1.58381915e-01]
     [5.85832676e-01 4.69261970e-02 4.49434847e-01 3.93760601e-01
      7.89355117e-01 3.53199694e-01 5.72347779e-01 6.01739135e-01
      1.67563696e-01 3.99907027e-01 5.45776104e-01 6.30217635e-01
      1.30219306e-01 9.07390217e-01 7.91456584e-01 3.44962215e-01
      6.70328327e-01 1.27873375e-02 1.19891384e-01 4.08509191e-01
      9.61505654e-01 2.06156903e-01 7.16620516e-01 7.87112304e-01
      4.07001026e-01 8.01709504e-02 1.35851157e-01 3.59771384e-01
      4.19368031e-01 1.76208130e-01 5.47808353e-01 8.94358692e-01
      9.68440325e-01 5.96414556e-02 6.53637163e-01 8.10309288e-01
      9.82330863e-01 7.75340636e-01 6.87218708e-01 5.58430316e-01
      3.65192408e-01 3.48994339e-01 6.80945362e-01 1.93069374e-01
      1.18215972e-01 1.08832242e-01 1.43459333e-01 5.69379925e-01
      1.56327091e-01 1.21767518e-01]
     [2.24502844e-01 9.48486015e-01 3.32807718e-01 9.81136468e-01
      4.63648396e-01 8.30050425e-01 4.72812349e-01 7.52662178e-01
      3.69170471e-01 2.40943428e-01 9.36711205e-01 3.55939723e-01
      4.91873493e-01 1.42478978e-01 5.34181068e-01 1.28885039e-01
      3.65678160e-01 7.23522758e-01 6.75134163e-01 5.77673421e-01
      5.44011638e-01 5.06741244e-01 4.60582211e-01 9.49955718e-01
      1.82737754e-01 1.88716047e-02 1.70006544e-01 2.63572285e-01
      5.20921547e-01 3.38468366e-01 5.44394611e-01 6.92905927e-01
      4.58719679e-01 2.46288435e-01 8.70814439e-01 8.87593144e-01
      6.24917117e-02 7.18742286e-01 3.51460287e-02 9.89762660e-01
      1.67179022e-01 9.05753274e-01 7.47057685e-01 9.68457248e-01
      1.80235184e-01 5.53476121e-01 1.56274503e-01 1.53031705e-01
      7.77363834e-01 3.57955720e-01]
     [6.44103448e-01 6.10652955e-01 8.61528482e-01 1.86192132e-01
      7.57056675e-01 6.25917972e-03 8.88997645e-01 5.60190295e-01
      7.78102085e-01 7.68879259e-01 5.08866976e-01 4.14625899e-01
      7.45531934e-01 9.42978416e-01 5.15441545e-02 5.07498738e-01
      6.22196385e-01 1.31233597e-01 1.77969975e-01 1.76071555e-04
      3.05662794e-01 7.32209704e-01 8.23953013e-01 4.67618791e-01
      5.97785255e-01 5.16311088e-01 7.33762585e-02 1.01120547e-01
      2.68163995e-01 5.35862790e-01 7.80980015e-01 6.01414149e-01
      5.43838023e-02 8.22605155e-02 2.86067059e-01 3.22367166e-01
      2.34773916e-01 5.81265042e-01 2.83901549e-01 8.08501149e-01
      2.50157001e-01 3.98508576e-02 4.29980951e-01 1.94903927e-01
      8.65610184e-01 7.02643108e-01 9.30058975e-02 5.80190079e-01
      6.64651633e-01 5.83821269e-01]
     [2.26297379e-01 1.90022710e-01 9.53720667e-01 5.14270734e-01
      4.15346281e-01 4.96314544e-01 4.52366417e-01 5.94662560e-01
      7.91297007e-01 2.07625687e-01 5.63166422e-01 6.56349961e-01
      4.71712611e-01 2.75534331e-01 8.49809005e-01 1.56118581e-01
      2.13215209e-01 3.06627861e-01 3.71058907e-01 5.62302991e-01
      3.17328855e-01 8.73208874e-01 4.09203959e-01 5.71581260e-01
      3.17169135e-01 4.05446568e-01 7.96942267e-01 1.56060445e-01
      4.42942172e-01 8.01147077e-01 2.59941549e-03 8.08003680e-01
      4.39053569e-01 8.64538148e-01 9.08317809e-01 5.42054866e-01
      4.30537684e-01 8.00627252e-01 7.02198618e-01 9.07857650e-01
      9.64071352e-01 4.16449229e-01 2.17517449e-01 4.80606284e-01
      7.12293354e-01 1.06732510e-02 3.10536380e-01 9.22633818e-01
      6.81666189e-01 6.24167804e-01]
     [6.79900529e-01 1.75375765e-01 7.74831103e-01 4.51360411e-01
      7.84686889e-01 4.83854048e-01 4.53583053e-01 8.99983658e-01
      2.90939227e-02 2.03570207e-01 7.20679809e-01 9.34854654e-01
      8.99149364e-01 5.63587400e-02 5.63211048e-01 4.87033109e-01
      2.75330136e-01 8.95884694e-01 5.37172601e-01 8.80532007e-01
      1.41171486e-01 8.01480329e-01 9.00111518e-01 8.74942736e-01
      8.49791237e-01 3.99779403e-01 8.44924218e-01 1.22134667e-01
      6.65183025e-01 7.30792079e-01 9.24981066e-01 5.96340101e-01
      1.71169951e-01 7.81357782e-01 2.59912011e-01 3.27655470e-01
      6.74396196e-01 4.09835467e-01 6.23541561e-01 4.34381794e-01
      1.81901357e-01 6.96795979e-01 9.98273369e-02 3.90759757e-01
      4.02720018e-01 9.43538466e-01 9.40825161e-01 2.56271182e-01
      2.92165169e-01 4.26202172e-01]
     [2.69545542e-01 5.61694184e-01 8.03679368e-01 1.64893844e-01
      3.95601910e-01 2.91093083e-01 5.94307198e-01 1.22809162e-01
      6.00669507e-01 9.46186358e-01 4.89280074e-02 8.65981275e-01
      6.11723212e-01 2.43525055e-01 7.93425197e-01 8.79930422e-01
      9.30703814e-01 3.95002001e-01 2.80047450e-01 5.59755256e-01
      7.19201792e-01 3.20396194e-01 8.10228635e-02 2.77217469e-01
      6.98872573e-01 2.34750614e-01 2.76818363e-01 2.43105286e-02
      3.98969179e-01 5.85609177e-01 4.99157641e-02 3.78621719e-01
      2.11882347e-01 1.02583768e-01 7.82066037e-01 3.53437154e-01
      7.06462578e-01 4.41651121e-01 9.32809328e-01 9.50072870e-01
      4.55079788e-01 3.10588072e-02 2.58934505e-01 2.18415285e-01
      2.81612870e-02 7.35993055e-01 7.92541552e-01 9.59728281e-01
      1.98256211e-01 7.02334701e-01]
     [1.61601982e-01 8.22810875e-01 8.57382420e-01 8.50873386e-01
      1.75159180e-01 9.93365268e-01 4.58487116e-01 8.56046915e-01
      2.09695512e-01 2.06523355e-01 4.45189109e-01 4.63978499e-01
      6.78257832e-01 3.60323355e-02 1.83188961e-02 2.13447350e-01
      4.70807303e-01 5.86784970e-01 7.49137860e-01 7.98306992e-01
      7.44758307e-01 9.60715199e-01 9.09112152e-01 2.35451493e-01
      2.95013959e-01 9.07803306e-01 4.43469907e-01 5.96816275e-01
      9.08894465e-01 2.26281771e-01 7.91623177e-01 6.31917572e-01
      4.46217661e-01 8.21809641e-01 8.38562071e-01 4.53824709e-01
      9.92758966e-01 7.32596228e-01 3.26920666e-01 7.36156045e-01
      1.12227943e-01 6.61665942e-01 1.71657664e-01 6.45049338e-01
      8.64207521e-01 7.88614831e-01 9.55863205e-01 2.17488735e-01
      1.87450463e-01 9.99267725e-01]
     [6.77134178e-02 7.98943004e-01 7.60731941e-02 4.31534582e-02
      5.04917078e-01 7.91571716e-01 6.64867343e-01 5.60482103e-01
      5.91621164e-01 5.10131195e-01 3.06537241e-01 8.41231744e-01
      9.74857525e-01 8.89182445e-02 5.00240548e-02 2.60717570e-01
      8.53998697e-01 4.61732505e-01 6.32170213e-01 5.55705402e-01
      3.55734775e-01 8.03635509e-01 7.32915671e-01 6.46466484e-01
      3.76342040e-01 5.28209875e-01 5.57619492e-01 2.94673140e-01
      7.00268348e-01 4.62178177e-01 9.39428117e-01 7.63576114e-01
      1.73729812e-01 1.88256063e-01 8.53717168e-02 2.21594964e-01
      1.83469270e-01 7.26077403e-01 3.04076658e-01 6.95961183e-02
      7.00311650e-01 8.28326598e-01 1.42210571e-01 2.66220893e-01
      4.71491960e-01 1.71255982e-01 7.60325963e-01 7.99322264e-01
      8.98345823e-01 4.69914376e-03]
     [1.60392551e-01 5.41962693e-01 5.80370963e-01 6.33619092e-01
      7.96169087e-01 4.19931270e-01 3.49820045e-01 8.58232840e-01
      6.27988163e-01 7.82169975e-01 9.02547620e-01 7.05029466e-02
      4.07994048e-01 4.92399171e-01 1.61680470e-01 4.79936532e-01
      3.15943925e-01 7.48017998e-01 4.43633150e-02 6.60536663e-01
      7.78206940e-02 5.89816935e-01 2.06733124e-02 5.65564014e-01
      9.99321781e-01 6.59645779e-01 7.63963049e-01 8.40377792e-01
      6.15912592e-02 5.93778956e-01 8.74509137e-01 1.82305575e-01
      8.95003312e-01 2.33764797e-01 7.04413553e-01 3.45911600e-01
      9.77750522e-01 6.14799922e-01 8.07888004e-01 5.19101916e-01
      3.19441432e-01 5.93872905e-01 1.75834975e-01 2.24650235e-01
      2.01086076e-01 3.80821750e-01 7.30454687e-02 1.41431801e-01
      6.69907037e-01 7.40191359e-02]
     [3.30593859e-01 7.40289231e-01 7.98166307e-01 7.83321444e-01
      8.65575729e-01 2.98719822e-01 1.58527585e-01 7.08109818e-01
      6.80408612e-01 7.35117242e-01 9.43761181e-01 7.67959137e-01
      9.97013241e-01 3.67101011e-01 7.63258600e-01 7.45133816e-01
      5.30847187e-01 4.74846305e-01 4.06200673e-01 7.80749890e-01
      6.63707802e-01 2.60748212e-01 8.67788019e-01 8.54440487e-01
      1.44999959e-02 7.09314964e-01 1.88188663e-01 2.58823709e-01
      7.95493701e-01 8.26785449e-01 2.77295828e-01 2.86020608e-01
      4.12981863e-01 2.07092008e-02 7.68729594e-01 9.45194355e-01
      7.38292744e-03 4.23737220e-01 6.31703107e-01 4.46821555e-01
      3.62170232e-01 1.91218924e-01 1.22643423e-01 8.21479234e-01
      9.02215499e-01 1.84815071e-01 4.01174243e-01 3.14879922e-01
      3.77497100e-01 7.44965562e-01]
     [7.77053921e-01 2.35419489e-01 7.09578833e-01 2.21537035e-01
      5.01335063e-01 7.11603835e-01 5.94046607e-01 4.64221570e-01
      8.17429579e-01 3.14442202e-01 9.27260889e-01 3.20730930e-01
      2.75314918e-01 1.85956643e-01 2.58579123e-01 9.22509013e-01
      9.14877648e-01 4.40313934e-01 5.87450265e-01 5.94797055e-01
      4.13738355e-01 1.98652500e-01 4.41664266e-01 2.88317727e-02
      7.91974361e-01 4.29381240e-01 8.64920703e-01 2.88044648e-01
      6.33779159e-01 8.84159319e-01 5.99882141e-01 6.53870277e-01
      8.37018884e-01 1.36643176e-02 4.74381645e-01 8.03322656e-01
      5.39102576e-01 2.24950397e-01 2.52444626e-01 3.50411071e-01
      6.78879308e-01 2.56271077e-01 6.51230836e-01 5.57034577e-02
      3.48867799e-01 8.33588663e-01 6.38824005e-01 7.99137490e-01
      4.37760196e-01 3.55883574e-01]
     [6.84361830e-01 3.16652828e-01 1.97257899e-01 1.26678784e-01
      7.60471767e-02 3.99468039e-02 6.23442327e-01 8.99956097e-01
      2.91457866e-01 5.76907513e-01 7.60338575e-01 7.31439993e-01
      1.15080243e-01 8.88226949e-01 6.87801993e-01 4.01423651e-01
      2.73495184e-01 4.71115420e-01 4.06356182e-01 7.89409613e-01
      1.23422717e-01 4.66277057e-02 5.98017748e-01 2.95715418e-01
      1.31222257e-01 8.33206169e-01 8.70153102e-01 8.39841716e-01
      8.90747661e-01 6.43582815e-01 9.26239058e-01 2.58899402e-01
      4.60422794e-01 8.80037888e-01 2.53905685e-01 6.17429522e-01
      3.13637282e-01 9.77377841e-03 3.97574982e-02 6.21990477e-01
      8.91024606e-01 2.87192945e-01 6.01163842e-01 7.90091509e-01
      3.62065478e-01 9.00887675e-01 3.51150935e-01 7.25035058e-01
      5.47455992e-01 8.93993085e-02]
     [1.41617724e-01 3.26927062e-01 7.37450230e-01 8.99632920e-01
      5.17782844e-01 5.48095240e-01 5.76335935e-02 1.47010595e-01
      5.31332433e-01 9.05799929e-01 5.20380505e-01 3.14046879e-01
      5.45301668e-01 2.79731045e-01 3.58299592e-01 8.79768424e-01
      3.77795667e-01 8.43499989e-01 1.61886591e-01 7.41574283e-01
      9.99107495e-02 9.96665026e-01 9.86857781e-01 3.50404680e-01
      5.87686635e-01 5.88893606e-01 5.24143412e-01 7.57274731e-01
      6.41220915e-01 5.35189031e-01 9.47234170e-01 5.60587349e-01
      4.37504001e-01 1.98195787e-01 2.21247270e-01 4.61333812e-01
      1.63166363e-01 1.46015916e-01 6.87412450e-01 2.42748813e-02
      2.81201528e-01 8.60424795e-01 7.51798637e-01 6.86093905e-01
      3.59895278e-01 7.71966856e-03 8.78946826e-01 3.96973299e-02
      1.34644385e-01 1.48922696e-01]
     [4.83211612e-01 2.11342776e-01 3.89103437e-01 4.88028332e-01
      6.33244344e-01 3.19890221e-01 8.86224029e-01 7.81489121e-01
      7.16106184e-01 9.47798828e-01 7.61932851e-01 8.89782948e-01
      1.52267166e-01 6.06834899e-02 1.32222714e-01 6.34730241e-01
      6.53190612e-01 6.31511469e-02 5.90230543e-01 7.13097156e-01
      2.84799306e-01 4.08583489e-01 3.88959627e-01 2.49390621e-01
      6.88725139e-02 6.93363158e-01 4.35914765e-01 7.22825493e-01
      6.12449813e-01 8.67747596e-01 5.80395211e-01 3.71339637e-01
      1.01786572e-01 5.73240918e-01 7.72163484e-01 5.21016457e-01
      3.22519750e-01 6.29818015e-01 6.10760201e-01 4.92553435e-01
      2.59542722e-01 9.16245923e-01 4.22026093e-01 6.77235848e-01
      1.91199895e-01 2.84449660e-01 4.34501782e-01 2.30775641e-01
      5.56471459e-01 2.03636233e-01]
     [5.28042313e-01 4.11320878e-01 5.43472397e-01 7.82162323e-01
      4.01639241e-01 8.52040787e-02 5.81047802e-01 4.90939472e-01
      5.97261127e-01 3.51904431e-01 3.88029398e-01 5.42341471e-01
      8.31136123e-01 6.98232876e-01 4.24420813e-01 5.55977688e-01
      2.42794927e-01 7.98560723e-01 2.19193906e-02 6.68728955e-01
      8.92382350e-01 2.25810722e-01 4.32944238e-01 9.77840962e-01
      5.84776466e-01 3.47453813e-01 3.56642581e-01 5.21073768e-01
      2.99407001e-01 5.68581105e-02 1.28260341e-02 7.82262515e-01
      2.89734652e-02 2.88041194e-01 4.19198347e-01 7.62481881e-01
      5.92376405e-01 7.26227339e-01 7.86324893e-01 2.09136571e-02
      3.52724702e-01 7.34158623e-01 2.53742494e-01 6.25225634e-01
      5.11872721e-01 2.27319814e-01 1.93031813e-01 4.44715423e-01
      1.26344445e-01 6.18300856e-01]
     [3.36575372e-02 3.28284806e-01 1.74337979e-01 1.96717856e-01
      6.06402568e-01 1.99231570e-02 9.46651758e-01 7.90928566e-01
      4.91134933e-01 7.26075816e-01 8.49137811e-02 2.11978397e-01
      1.95695004e-01 3.58949207e-01 1.46670137e-01 4.55862629e-01
      9.90806079e-01 7.25271839e-01 7.13620755e-01 5.99640169e-01
      3.04706807e-01 9.58247522e-01 1.62534947e-01 8.72165707e-01
      1.85745176e-01 1.62066991e-02 6.61045176e-01 9.23840001e-01
      7.17915734e-01 4.36930158e-01 5.61376219e-01 6.98545993e-01
      9.16597884e-01 4.02984117e-01 9.13172538e-01 7.81888460e-01
      3.99686245e-01 2.08217306e-01 7.49743174e-01 8.88932245e-01
      5.17586965e-01 6.42383421e-01 4.34892984e-01 1.12229419e-01
      7.10137390e-01 9.20241124e-01 5.25063245e-02 2.14008777e-01
      1.45656974e-01 5.89479743e-01]
     [6.60335812e-01 5.90024828e-01 6.49940848e-01 7.21829904e-01
      1.37624774e-01 8.27112220e-01 8.90654973e-03 6.78435553e-01
      7.13581431e-01 7.73961495e-01 7.86381640e-01 1.40685484e-01
      6.83692065e-02 7.40419628e-01 9.42222499e-01 2.29840685e-01
      8.83614300e-01 8.54835525e-01 2.73841666e-01 1.29301345e-01
      8.97160103e-01 4.04563706e-01 1.07817499e-01 9.83734260e-01
      1.81653598e-01 6.35045672e-01 6.14803012e-01 6.27673659e-01
      6.78672764e-01 1.03498762e-01 4.16057356e-02 9.49273813e-01
      7.41503765e-01 2.20029735e-01 3.86060612e-01 7.84661497e-01
      3.69850254e-02 9.97613212e-01 5.87726663e-01 2.88472957e-02
      9.80929305e-01 1.44050606e-01 6.51122301e-01 5.19613285e-01
      7.15414651e-01 7.48279013e-01 3.50958714e-01 5.16836088e-01
      1.71377722e-01 1.61327213e-01]
     [4.12075595e-01 2.17900699e-01 3.55076707e-01 9.59830519e-01
      5.47232274e-02 2.23339089e-01 5.22411055e-01 9.65150110e-01
      7.41732391e-01 3.04323043e-01 8.93653400e-01 2.37356917e-01
      8.38045436e-01 5.89849595e-01 7.46249821e-01 1.70236936e-01
      2.81571761e-01 9.95551894e-02 8.00219149e-01 7.31865913e-01
      9.58400083e-01 3.80146512e-01 3.80116585e-01 9.62082322e-01
      6.32716028e-01 8.84041340e-01 1.42174107e-01 4.08699286e-01
      4.82625234e-01 3.11603529e-01 2.10748967e-01 5.26729960e-01
      1.70571041e-01 7.88818064e-01 1.30365418e-01 1.36499916e-01
      1.30308219e-01 4.26193251e-01 1.47779766e-01 5.57586180e-01
      1.62465069e-01 5.10190670e-02 1.13472925e-01 5.25956737e-01
      1.97359722e-01 3.98233236e-01 7.52937170e-01 9.42408671e-01
      2.39461753e-01 8.10087155e-01]
     [8.98743796e-02 3.93319418e-01 8.67469763e-01 8.47044040e-01
      8.47435319e-01 1.17176908e-01 3.36886972e-01 8.41152881e-01
      3.11858411e-01 2.34784646e-01 4.24213940e-01 4.12674845e-01
      3.73540059e-01 6.62450157e-01 3.65885620e-01 7.41667896e-01
      7.11859435e-02 3.19276686e-01 9.90213054e-01 8.46348955e-01
      7.07469654e-01 5.47804787e-01 8.45357140e-01 9.19110201e-01
      3.91634374e-01 9.82410034e-01 7.44038954e-01 8.27533975e-01
      3.56214080e-01 7.76939575e-01 4.86318543e-01 3.26370047e-01
      4.31871367e-01 7.44016852e-01 4.16789099e-01 4.09098266e-01
      7.46575360e-01 5.62792261e-02 6.57659504e-02 7.88813467e-01
      1.54101913e-01 3.97015843e-01 8.93668828e-01 7.10012471e-01
      3.58170535e-01 4.53257981e-01 1.32801695e-01 1.53608207e-01
      6.43725286e-01 2.39397302e-01]
     [5.92478371e-01 7.10386573e-02 6.43824346e-01 4.37153586e-01
      5.50041803e-01 7.32301382e-01 7.80527238e-01 3.12694816e-01
      2.48292673e-01 5.99210765e-01 5.43997835e-01 4.24233870e-01
      1.66778828e-01 5.13229559e-01 8.07090311e-01 2.74487079e-01
      5.34546731e-01 8.29036687e-01 6.86931585e-01 2.93729122e-01
      7.06311706e-01 6.57466558e-01 6.13369677e-01 8.61775390e-01
      2.40748354e-01 3.39624568e-01 2.78622635e-01 7.17970414e-02
      9.90659117e-01 2.81590797e-01 3.66397331e-01 2.14078646e-01
      7.42824051e-01 1.35995891e-01 4.19733608e-02 8.79760595e-01
      1.67987788e-01 3.54636797e-01 3.10008705e-01 3.62871222e-01
      5.04966029e-01 4.84380573e-01 7.95457248e-02 4.20032554e-01
      2.13480489e-02 5.37209984e-02 1.32871020e-01 1.92958663e-01
      9.54114448e-01 4.04372340e-01]
     [8.05879745e-01 9.02487545e-01 5.21292164e-01 1.22248519e-01
      3.93834242e-01 8.97451946e-01 1.26920820e-02 9.63549754e-01
      6.48947841e-01 5.82222807e-01 6.46400557e-01 7.78306066e-01
      6.11904887e-01 1.34928446e-01 9.14806380e-01 3.11983886e-01
      9.68564057e-01 2.24152252e-01 8.30691840e-01 3.68626664e-01
      9.20400704e-01 1.89308914e-01 4.32956656e-01 9.90321701e-01
      3.58273162e-01 7.10955101e-01 2.69503379e-01 9.37914984e-01
      5.93787966e-01 4.30482244e-01 9.66779309e-01 4.62536889e-01
      8.28494133e-01 3.60502875e-01 9.30781040e-01 5.27588299e-01
      3.81180689e-01 7.71739635e-01 3.71208160e-01 1.55923174e-01
      4.90518222e-01 7.80761612e-01 4.84916383e-01 8.72735794e-01
      2.35262208e-01 2.39182303e-02 1.70430238e-01 2.58213049e-01
      7.42570981e-01 8.57315083e-01]
     [3.78637004e-01 7.06029925e-01 3.67714518e-02 2.91109303e-01
      8.25940750e-01 2.96454524e-01 7.40251167e-01 9.75473325e-01
      6.01355530e-01 9.59431582e-01 5.08018680e-01 7.45247439e-01
      1.48648806e-01 5.26337365e-02 8.62423675e-01 9.12985188e-01
      4.16586737e-01 1.55633370e-01 8.12244714e-01 4.94405258e-01
      5.88303573e-01 5.84922648e-01 5.73994204e-01 3.16627542e-01
      1.64149496e-02 4.77506196e-01 9.53456486e-01 1.83976006e-01
      5.99057154e-01 1.27598759e-01 8.54932136e-01 1.33327996e-01
      7.25956077e-01 2.67633363e-01 1.85201179e-01 1.52416410e-01
      5.23768657e-01 5.18796429e-01 2.51048248e-01 7.45534081e-01
      7.79121647e-01 5.69040139e-01 7.68762413e-01 6.21715699e-01
      4.67321421e-01 7.82091490e-01 9.18734500e-01 3.21732444e-01
      7.70242133e-01 5.61629744e-01]
     [3.26231074e-01 2.86861910e-01 5.68731038e-01 5.23030448e-01
      6.93229991e-01 1.86302847e-01 1.02598615e-01 7.82906691e-01
      2.25924779e-01 6.95240972e-01 2.28906149e-01 1.71802263e-01
      2.00421378e-01 7.05552421e-01 8.67238656e-01 5.74599352e-01
      2.88871884e-01 4.75244802e-01 5.65055068e-01 4.72919972e-01
      5.68022017e-01 3.10222669e-01 6.07368176e-01 9.97920264e-01
      2.12876986e-02 7.96269926e-01 3.06637682e-01 5.44683767e-01
      8.13148721e-01 6.02823428e-01 9.83427058e-01 9.29673840e-01
      1.95066438e-01 6.20013585e-01 4.22968015e-01 6.79969793e-01
      2.45124465e-01 8.35467967e-01 5.04620537e-01 5.48662464e-01
      1.29728716e-01 8.07737242e-01 3.07115715e-01 2.78329013e-01
      5.23942105e-01 1.29304757e-01 1.86638057e-01 8.69679338e-01
      7.98436451e-01 6.55591732e-01]
     [4.82511809e-01 2.66054600e-01 2.42780615e-02 3.53509827e-01
      7.97089900e-03 2.36605642e-01 2.65821101e-01 5.97227352e-01
      7.84152372e-01 9.17176006e-01 6.96059567e-01 5.61981345e-01
      2.77069620e-01 4.58277801e-01 5.42093325e-02 9.11540275e-01
      9.82658680e-02 2.58616683e-01 1.15678338e-01 6.44295110e-01
      1.41292658e-01 9.55339681e-01 9.18835649e-01 1.80581786e-01
      6.54665249e-01 4.21558516e-01 8.27160856e-01 6.16151805e-01
      1.80496016e-01 7.96135434e-01 7.92771500e-01 8.39724201e-01
      2.26053672e-01 7.18529337e-01 8.01843393e-01 3.27201526e-01
      3.42405196e-03 2.64285076e-01 4.18077968e-01 5.58529030e-01
      1.20624091e-01 1.88947971e-01 9.51877899e-01 9.14027997e-01
      2.60244873e-01 5.87403742e-01 6.73820421e-01 7.67253443e-01
      2.01166371e-01 1.34691702e-01]
     [6.49530972e-01 5.02199364e-02 2.53575261e-01 2.74647194e-01
      1.58952376e-01 7.24724109e-01 8.95431085e-01 6.87151658e-01
      8.63698938e-01 6.78888018e-01 8.64306578e-01 4.14001212e-02
      7.71836190e-01 5.06367961e-01 5.08737247e-01 9.04501193e-01
      1.76456317e-01 1.18610668e-02 1.63570143e-01 1.90361250e-01
      1.23885101e-01 3.48241664e-01 3.77872706e-01 6.52316468e-01
      6.32991823e-01 3.51716520e-01 7.15706253e-01 9.89268377e-01
      6.62415154e-01 5.73149141e-01 7.70069694e-01 5.44650782e-01
      5.13901352e-01 7.70927542e-01 8.19342672e-01 9.22794671e-01
      5.69373660e-02 2.27219361e-01 3.79948534e-01 6.82911579e-01
      1.21980135e-01 5.78915187e-01 9.97900298e-02 3.23915824e-02
      5.50844764e-03 8.72689839e-01 3.29136229e-01 7.92236161e-02
      9.38884842e-01 7.38260139e-01]
     [4.94360428e-01 1.03558746e-01 3.17316527e-02 3.05614713e-01
      2.66055152e-01 7.05146366e-01 4.55481800e-01 9.66495342e-01
      7.93175837e-01 7.37773406e-02 2.72047360e-01 7.87452036e-01
      5.21251506e-01 2.94650040e-01 2.82343610e-01 8.13731391e-01
      5.53945065e-01 4.23775249e-01 3.51794516e-01 7.20660209e-01
      7.74859009e-02 9.31206333e-01 3.97680817e-01 6.51535472e-01
      4.19334384e-01 3.65948638e-01 7.31140806e-01 3.43629156e-01
      9.00930793e-02 5.80093651e-02 5.70139457e-01 9.02487063e-01
      8.90036056e-03 1.17754995e-01 2.98489693e-01 3.14390103e-02
      7.94638114e-01 7.30042223e-01 3.70064123e-01 7.86345453e-01
      4.85150573e-01 7.18064393e-01 2.17741755e-01 1.40080478e-03
      9.31104308e-01 4.98848708e-01 4.78951322e-02 6.33677802e-01
      3.50582735e-01 7.58812339e-01]
     [6.64298180e-01 4.48134136e-01 5.40221728e-02 4.15713827e-03
      4.73529144e-02 3.24834272e-01 1.48990508e-01 5.98453323e-01
      3.51049310e-01 1.47205752e-01 5.87700912e-01 7.29046135e-01
      4.68816832e-01 8.18945463e-02 1.26726252e-01 5.74167470e-01
      9.40781132e-01 4.49903491e-01 2.24380925e-01 3.10749826e-01
      8.04094268e-01 3.09511112e-01 4.57434275e-01 3.75617563e-01
      5.71064562e-02 5.50079558e-01 4.44212104e-01 7.77155064e-01
      2.72738179e-01 3.50739275e-01 6.50776902e-01 8.87102060e-01
      3.93688738e-01 7.46466066e-01 9.27755859e-01 5.22213051e-01
      3.71458513e-01 5.87165920e-01 6.58633973e-01 8.65281658e-01
      8.89576662e-01 3.23110054e-01 3.64237399e-01 2.69940448e-01
      5.79820990e-01 4.75327762e-01 8.94432652e-01 7.78382183e-02
      6.94581555e-01 4.83675577e-01]
     [5.32973180e-01 8.92026344e-01 5.05390928e-02 1.68257752e-01
      8.93138785e-01 9.21862785e-01 9.59631844e-01 7.80442489e-01
      4.75536322e-01 1.28115503e-01 9.77614409e-01 8.07821346e-01
      1.79455263e-01 2.49231464e-01 7.30633421e-01 6.49643453e-01
      5.28831188e-02 6.56147442e-01 7.74642467e-01 6.25918669e-01
      1.36226100e-01 6.02651503e-01 5.88287596e-01 7.14673404e-01
      1.87829972e-01 7.53472814e-01 3.12259089e-01 7.66906837e-01
      9.24047973e-01 1.12139627e-01 7.34354219e-01 8.02892261e-01
      9.63782968e-02 2.71216198e-01 2.89517519e-01 8.52599127e-01
      1.62309792e-01 2.88365281e-01 7.86824469e-01 6.05109914e-01
      4.58678077e-01 6.42330101e-01 5.47335284e-01 3.14345921e-01
      1.65936483e-01 3.01871238e-01 8.44254222e-01 6.41537396e-01
      1.66742060e-01 7.54753939e-01]
     [8.19768572e-01 1.24943677e-01 2.33175491e-01 1.91034915e-02
      1.33178916e-01 2.52742429e-01 5.12430412e-01 3.10073922e-01
      8.36689379e-01 6.15859211e-01 4.79470686e-01 2.87390564e-01
      4.55271037e-02 9.21188119e-01 3.55621070e-01 8.81364052e-01
      5.19543280e-01 2.91936517e-01 6.31556934e-01 4.22156580e-01
      8.33871994e-01 1.87318396e-01 7.48329444e-01 4.16517801e-03
      4.34322368e-02 9.73960201e-01 6.08745251e-01 1.62335127e-01
      9.09819173e-02 1.03452739e-01 8.52554333e-01 7.94041491e-02
      2.44723333e-01 1.94306582e-01 7.09629356e-01 5.67432247e-01
      3.99027958e-01 3.62690709e-01 7.93570915e-01 2.28403051e-01
      4.42390752e-01 3.52210599e-01 9.57329078e-01 1.36428313e-01
      2.86334427e-02 5.79196322e-01 6.16004787e-01 1.79844125e-01
      7.97899272e-02 8.89127390e-01]
     [9.48870138e-01 1.76315058e-01 1.46754297e-01 7.36010899e-01
      6.21041269e-01 1.07500960e-01 2.20622360e-01 6.07546784e-01
      7.35378754e-01 7.80417336e-01 5.00510630e-01 7.99200179e-01
      5.27987010e-01 9.62864967e-01 5.43792860e-01 8.56957927e-01
      9.82819439e-01 1.83754925e-01 3.72543131e-01 4.56799610e-01
      2.41910631e-01 4.75922252e-02 8.83353524e-01 6.42575544e-01
      4.49106621e-01 6.17295538e-01 6.03524935e-01 8.13733762e-01
      2.47428830e-01 9.54121051e-01 7.07907705e-01 6.82999229e-01
      3.68114641e-01 2.57441549e-01 7.04024133e-01 9.00871569e-01
      1.67427942e-01 6.78761324e-01 2.09743074e-01 5.81888362e-01
      5.80884146e-01 4.07756745e-01 8.93696322e-01 8.14044484e-01
      1.06057808e-01 5.64546123e-01 9.06198657e-01 3.91225943e-01
      8.01672827e-01 2.38335075e-01]
     [8.59950980e-01 1.85091534e-01 2.82288353e-01 2.46907229e-01
      7.27067686e-01 2.11703243e-02 1.10021325e-03 5.84172702e-01
      8.75798660e-01 1.06481615e-01 9.14070810e-02 2.09663148e-01
      9.48718962e-01 8.13095052e-01 7.26726469e-01 8.97068581e-01
      3.83157624e-01 4.81488043e-01 7.68767675e-01 9.91773247e-01
      2.35335829e-01 8.91896723e-01 6.16467244e-01 4.09699981e-01
      3.82360927e-01 4.48453799e-01 2.10999381e-01 8.86483065e-01
      7.67044279e-01 7.31271321e-01 3.16819439e-01 1.82101537e-01
      7.14779065e-01 4.28881562e-01 5.11690483e-01 8.00094793e-03
      6.78340634e-01 5.78531504e-01 3.14011038e-01 4.41427767e-02
      5.66721437e-01 9.53193650e-01 7.32164846e-01 6.49544208e-02
      8.50542953e-01 9.62424273e-01 5.02664136e-01 5.51869993e-01
      7.39756995e-01 6.73593132e-01]
     [2.21468371e-01 6.82529696e-01 1.54419941e-01 7.47373026e-01
      4.26441027e-01 2.25843555e-01 5.19570558e-01 9.77858836e-01
      2.75422599e-01 2.47171738e-01 8.81533096e-01 4.41846451e-01
      2.38266420e-01 6.64592957e-01 5.86150176e-01 8.34738962e-01
      6.86837733e-01 8.30071568e-01 7.54083032e-01 9.47480128e-02
      3.14677012e-01 4.12037967e-02 4.19330611e-01 1.05115647e-01
      5.10466490e-01 9.56043361e-02 6.03028321e-01 7.09911106e-02
      6.99281277e-01 8.10313539e-01 9.60674257e-01 6.39095295e-01
      9.40594104e-01 4.01508589e-01 9.80497538e-01 4.46246222e-01
      4.16778301e-01 4.76258657e-01 4.01213191e-02 2.00378857e-01
      3.14792794e-01 3.95707525e-01 2.76452600e-01 5.06984216e-01
      5.41419890e-01 2.25459226e-01 7.14342214e-01 4.65397314e-01
      4.54868978e-01 4.83640337e-01]
     [2.11723113e-01 1.18020937e-01 5.74394843e-01 8.69534374e-01
      4.82339342e-01 8.43486876e-01 9.09842802e-01 5.32612353e-01
      2.56363340e-01 7.92744118e-01 9.01600159e-01 1.80116772e-01
      5.35315472e-01 2.67447427e-01 2.46667470e-01 6.69541409e-01
      6.51101187e-02 4.90682956e-01 7.15824671e-01 3.58057461e-01
      5.55041258e-01 8.81707621e-01 4.77350057e-02 8.26420660e-01
      6.24315104e-01 3.57662328e-01 8.78390991e-01 5.24480601e-01
      6.83730700e-01 1.14068528e-01 4.40325932e-01 5.87051129e-01
      6.52122481e-01 8.63816990e-01 3.22548122e-01 3.97400424e-01
      6.03810585e-01 1.18216101e-01 2.13552987e-01 7.50158531e-01
      5.22924822e-01 4.36510532e-02 8.32066793e-01 4.45261122e-01
      7.06814188e-01 8.61820929e-02 7.82866668e-01 8.67835431e-02
      6.83776476e-01 8.54748189e-01]
     [7.28414023e-01 6.28000219e-01 6.07992183e-01 7.95909939e-01
      2.66618911e-01 9.79368309e-01 4.91486078e-02 1.79660821e-01
      2.59526006e-01 1.18172380e-02 8.56117776e-01 5.69047533e-01
      2.26618149e-01 4.59743669e-01 2.49983211e-01 1.98335469e-01
      5.02006590e-01 2.32434506e-01 7.52372617e-01 9.31146136e-01
      3.44611996e-01 3.16541541e-01 6.27970487e-01 2.05524782e-01
      9.09938556e-01 1.34475621e-01 9.57497818e-01 6.08814433e-02
      8.21203225e-01 6.77704908e-01 2.11604285e-01 7.47502293e-01
      5.36488317e-01 6.68349651e-01 4.15808559e-01 4.50734776e-01
      6.15595703e-01 9.19762327e-01 3.83088277e-01 6.39582719e-01
      4.94882339e-01 4.09994549e-01 3.37461967e-01 7.05128212e-01
      8.13299193e-01 2.52582290e-01 6.34228562e-01 4.21050816e-01
      6.01985950e-01 4.25845504e-01]
     [3.00968372e-03 2.54182752e-01 2.67143151e-01 6.27414651e-01
      9.63603434e-01 7.15642862e-01 2.40487949e-02 9.44879763e-01
      3.98285234e-01 4.67397788e-01 4.78829296e-01 4.32287674e-01
      6.84937482e-01 2.34680041e-01 1.48690614e-01 4.13198413e-01
      3.43646283e-01 2.29472247e-01 8.55011154e-01 6.18337534e-01
      6.94396050e-01 6.35106826e-01 1.19580524e-01 7.92073704e-01
      5.51422947e-02 4.66614657e-02 7.26222624e-01 1.45107096e-01
      3.18494351e-01 5.58340712e-02 2.70614024e-02 8.63075270e-01
      4.22202946e-02 5.28638108e-01 6.69593813e-01 8.35029062e-01
      2.55977306e-01 4.74062826e-01 6.41329127e-01 8.02311107e-01
      4.90889419e-01 1.36601777e-01 9.38832921e-01 1.32769206e-01
      6.17844823e-01 4.31022954e-01 5.09317906e-01 2.74034724e-01
      5.52861026e-01 1.04424839e-01]
     [5.09130840e-01 1.11191732e-01 5.69523846e-01 2.98006424e-01
      7.53137318e-01 7.74417545e-01 8.08076428e-01 1.25636263e-01
      4.17184228e-01 5.26947314e-01 2.21750371e-01 6.92734094e-01
      8.01542642e-01 2.54439792e-01 9.09707435e-01 1.46527661e-01
      2.41860567e-01 5.61130169e-01 9.89882381e-01 4.17289507e-01
      5.43849654e-01 5.19202113e-01 2.00671152e-01 2.31641014e-01
      9.99252808e-01 4.90781148e-01 1.28612614e-01 3.75219870e-01
      1.23694998e-01 4.72090334e-02 8.35462837e-01 1.84789770e-01
      5.56268233e-01 5.85902535e-01 2.72172863e-01 4.80343575e-01
      7.09086825e-01 8.44029814e-01 2.06129591e-01 8.00175425e-01
      5.30047394e-01 3.04658191e-01 6.34426412e-01 5.47884901e-01
      3.16001428e-01 5.07308928e-01 4.82682189e-01 3.33050513e-02
      8.79484407e-01 4.93907887e-01]
     [7.22887546e-01 4.60424458e-02 5.50645174e-02 5.72517616e-01
      6.30597302e-01 8.45909056e-01 4.49202020e-01 2.00334975e-01
      2.05563953e-01 7.11116964e-01 3.89918685e-02 4.51213943e-01
      5.93794167e-01 1.96395169e-01 7.75917144e-01 8.53433054e-01
      6.79200427e-01 2.49601221e-01 5.17664097e-01 1.26392566e-01
      1.90932852e-01 8.26646132e-01 4.16072368e-01 1.91565390e-01
      7.77555613e-02 5.91286329e-02 5.45122690e-01 8.31120273e-01
      9.03241874e-01 6.28707407e-01 9.68175002e-01 3.46013116e-01
      4.08853292e-01 7.61106795e-01 1.38843107e-01 5.38939794e-01
      9.22128641e-01 7.04794363e-01 7.22885594e-01 8.19896368e-01
      8.06882195e-01 8.88721656e-01 8.64242045e-01 8.83438565e-01
      3.21815489e-01 6.37086315e-01 6.80500611e-01 3.33667398e-02
      2.51296245e-01 3.97747281e-02]
     [6.80582558e-02 3.48867917e-01 5.12070701e-01 1.37063577e-01
      9.03886141e-01 8.24824456e-02 7.52606365e-01 9.14707682e-01
      1.11413113e-01 1.85768170e-01 1.46663118e-01 9.19234950e-01
      2.80589136e-02 1.77516043e-01 9.11692336e-02 4.02002145e-01
      1.59929375e-01 2.81453083e-01 6.19749877e-01 4.79425852e-01
      8.95315378e-01 7.92429361e-01 9.76826424e-01 9.76427853e-01
      5.33708622e-01 8.23082046e-01 9.02361247e-01 1.91448089e-01
      1.20352973e-01 6.99367457e-01 8.75767733e-02 9.31573904e-01
      6.66273641e-01 1.54517801e-01 5.06892853e-01 5.30220565e-01
      7.57355432e-01 1.96201957e-01 2.07069685e-01 2.10275671e-01
      4.46928701e-01 5.50209110e-01 3.14404087e-01 6.87559058e-01
      6.54896677e-01 8.50444679e-01 9.61750063e-01 6.50639936e-01
      4.01911200e-01 3.26131535e-01]
     [6.88868682e-01 7.35774371e-01 4.34528404e-01 8.22063175e-01
      2.08471452e-01 1.01168246e-01 1.46223616e-01 7.17441057e-01
      4.64033829e-01 1.87296476e-01 9.44689854e-01 6.40719311e-01
      1.47128662e-01 8.61825267e-01 7.95629160e-01 7.10037631e-01
      1.31723561e-01 3.01955364e-02 1.18713866e-01 5.75527383e-02
      5.75515571e-01 8.91998517e-01 5.40889715e-02 8.56228985e-01
      6.79087707e-01 9.57043632e-01 3.53060098e-01 3.18918693e-01
      8.48809784e-02 2.18214165e-01 1.98060079e-01 1.04404818e-01
      9.24642053e-01 6.86439816e-01 7.65896346e-02 5.74776833e-02
      3.46117595e-01 3.52735142e-01 5.70174849e-01 8.42072865e-01
      9.90374619e-01 2.85972629e-01 9.82916800e-01 2.29720739e-01
      4.96093379e-01 2.15388413e-01 2.09083317e-01 1.48193572e-02
      7.99991627e-01 6.18368062e-01]
     [2.14332756e-01 1.46565592e-01 6.74120677e-01 6.75158962e-01
      3.72233094e-01 9.70898072e-01 8.81900805e-01 2.60248982e-01
      1.07689016e-02 4.19663871e-01 4.15055546e-01 6.74237442e-01
      4.26250710e-01 7.70863234e-01 1.48543574e-01 7.97619993e-01
      5.61641266e-01 2.25665982e-01 2.00670835e-02 5.78143701e-01
      3.88555609e-01 2.67805971e-01 4.64043162e-03 8.77480891e-01
      3.85251983e-01 4.19603975e-01 5.96399156e-01 2.03756679e-01
      8.74071797e-01 2.44856006e-01 3.89017651e-01 6.41016669e-01
      3.86509118e-01 3.80314033e-01 6.50013243e-01 1.09622345e-01
      9.68253118e-02 5.96671841e-01 8.91136198e-01 7.19129201e-01
      9.82442482e-01 2.68437362e-01 3.88988530e-01 3.80945618e-01
      3.24215853e-02 7.01045416e-01 8.07348572e-01 3.09256355e-01
      1.10330628e-01 2.37757246e-02]
     [4.07314995e-01 6.23814313e-01 3.10657855e-01 9.69369228e-01
      8.54290382e-01 2.91514683e-01 3.26246266e-01 2.66657538e-01
      4.60254734e-01 1.66903378e-01 7.86095320e-02 9.96580655e-01
      5.80314630e-01 8.10559982e-01 2.09320607e-01 1.06288219e-01
      8.04168670e-01 1.26619728e-01 6.88982080e-01 6.03428951e-01
      3.37469298e-01 6.83154773e-01 1.24974366e-03 1.89481603e-01
      9.72252074e-01 1.62812432e-01 4.60942932e-01 9.31947987e-01
      8.59584029e-03 3.14000566e-01 3.92436545e-01 5.78641408e-01
      8.94039839e-01 3.22495814e-01 7.91799336e-01 7.56238398e-01
      8.05300495e-01 7.30900820e-01 4.37874156e-01 8.92822439e-01
      7.23897177e-01 6.42862955e-01 1.59368914e-01 8.53179250e-01
      5.12702228e-01 8.65738286e-01 1.63775543e-01 4.69664757e-01
      4.42050686e-01 2.40456324e-01]
     [1.89232974e-01 8.61294438e-01 9.23431526e-01 3.46840539e-01
      9.24594506e-01 7.26856721e-01 2.39028940e-01 3.39937237e-01
      3.58232128e-01 7.39567065e-01 3.75268339e-01 5.65542213e-01
      5.21416414e-01 4.01162074e-01 6.22820659e-01 3.12993908e-01
      2.87706538e-01 4.69333861e-01 8.98453202e-01 1.77696589e-01
      5.71609923e-01 1.09655719e-01 9.38571569e-01 9.65652703e-01
      8.94713113e-01 2.01789908e-01 8.73009889e-01 7.62406592e-01
      6.65265894e-01 3.71892007e-01 2.28206454e-02 4.99183914e-01
      6.08103989e-01 1.72847387e-01 2.44742045e-01 3.94994023e-01
      1.54988714e-01 7.00379325e-01 5.75788236e-01 9.83863750e-02
      8.14796283e-01 5.87117610e-02 3.44782078e-01 4.12823262e-01
      4.33315577e-01 2.41017871e-01 9.17566898e-01 5.31643031e-01
      8.50996943e-01 4.43364885e-01]
     [9.90482472e-01 7.78421569e-01 9.81286344e-01 6.53727808e-01
      3.68864184e-01 3.13389030e-01 2.90938577e-01 6.65767593e-01
      5.18884145e-01 4.35739663e-01 5.44034915e-01 3.37086711e-02
      7.92174804e-01 8.96716175e-01 7.57851729e-01 8.52732599e-01
      8.82971351e-02 7.55069403e-01 3.63760028e-01 8.22339351e-01
      1.16406692e-02 8.47633966e-01 2.64121574e-01 8.33342220e-01
      4.60009716e-01 1.84553485e-01 8.55387723e-01 8.17118238e-01
      4.17935251e-01 8.10935412e-01 5.56807746e-01 2.52673347e-01
      6.70030474e-01 5.95122148e-02 4.73671851e-01 4.33803282e-01
      1.41294002e-01 5.94169965e-02 7.15228743e-01 9.53522856e-02
      8.90669092e-01 9.12165426e-01 5.70257204e-01 2.99217474e-01
      3.47128470e-01 9.75168128e-01 1.34625923e-01 9.42069519e-01
      9.76001148e-01 1.51254238e-01]
     [4.03168054e-01 3.95690991e-01 4.90015916e-02 1.77316880e-01
      7.39750926e-01 4.91618402e-01 7.16056938e-01 8.48815021e-01
      5.99150158e-02 1.52799205e-01 2.16515417e-01 6.28681628e-01
      3.16947148e-01 9.19609929e-02 6.82589463e-01 3.88757498e-01
      4.23497697e-01 9.36054219e-01 2.63130301e-01 4.61320950e-01
      8.70614209e-01 8.15257059e-01 7.35991228e-01 7.27697175e-01
      9.04660373e-01 9.83861843e-01 4.99398750e-01 4.46526811e-01
      2.30547110e-01 6.13690243e-01 7.98055470e-01 6.51295593e-01
      2.94839448e-01 1.09887805e-02 2.80827236e-01 4.65303323e-01
      3.53858425e-01 8.83293197e-01 1.13647193e-01 2.53900762e-01
      4.06208541e-01 5.22090706e-01 3.71352528e-01 9.78161366e-01
      5.43947105e-01 7.41213168e-01 8.36391655e-02 8.65328843e-01
      9.43465721e-01 1.35656045e-01]
     [7.79004061e-01 6.97893617e-01 5.66305257e-01 6.27112500e-01
      2.94437708e-01 9.33351256e-01 3.83361437e-01 8.39088308e-01
      9.05291708e-01 4.37490654e-01 7.23489939e-01 5.04964350e-01
      8.92014614e-01 6.56463506e-01 6.58420491e-02 7.14017913e-01
      4.25072788e-01 1.00761092e-01 5.47661996e-01 6.81529843e-01
      5.44666616e-01 4.77917478e-01 1.96379283e-01 1.36731134e-01
      2.91615629e-01 2.70237439e-01 9.34585008e-01 4.33885969e-01
      5.14261936e-01 7.75462862e-01 7.08908495e-01 5.79824247e-01
      2.30575071e-01 4.86472608e-01 4.28397299e-01 6.55590007e-01
      4.60494966e-01 2.22568150e-01 3.60020503e-01 9.09277739e-02
      9.20812106e-01 9.84244022e-03 4.46825420e-02 6.17487261e-01
      8.09600854e-02 8.00100980e-01 2.88511312e-01 7.41628329e-01
      4.33634865e-01 5.35114849e-01]
     [6.26121371e-01 4.91955020e-01 5.87384693e-01 3.56949008e-02
      4.10578162e-01 6.68803989e-01 9.47424878e-01 3.46864689e-01
      3.89979756e-01 5.54772920e-01 9.14836293e-01 9.24236026e-03
      4.51421925e-01 5.53389329e-01 8.35928398e-02 1.87181451e-01
      9.64022648e-01 8.12808893e-01 7.25301861e-01 6.57469017e-01
      2.55638990e-01 5.14077141e-01 7.57308725e-01 4.36858356e-01
      5.75110579e-01 5.56969553e-01 2.60245449e-01 5.37752654e-01
      9.55558246e-01 6.19755793e-01 5.25941873e-01 5.96109822e-01
      5.72884008e-01 6.56443197e-03 2.14980056e-01 1.86122619e-01
      7.53267313e-01 7.19209356e-01 9.72341144e-01 4.48036759e-01
      5.07992571e-01 1.66642089e-02 5.45422435e-01 7.42916772e-01
      1.42832266e-01 5.41663915e-01 3.70217330e-01 1.23395860e-01
      7.14744522e-01 5.32380707e-01]
     [8.53060845e-01 8.89868357e-01 4.92233622e-01 9.21376624e-02
      3.32904750e-01 5.99252495e-01 8.35415400e-01 1.01449014e-01
      8.45199949e-02 8.28393410e-01 9.60338124e-01 8.70991096e-01
      9.49444725e-01 2.57270394e-01 5.96272335e-01 8.67143703e-01
      6.93525858e-01 3.89976918e-01 1.97752276e-01 3.51479336e-03
      1.98850959e-01 7.15290602e-01 9.46090263e-01 6.72586527e-01
      6.74767016e-01 9.35034671e-01 3.81249065e-01 5.70280153e-01
      1.10508919e-01 7.55895046e-01 5.81063332e-01 4.87136330e-01
      7.20992908e-01 2.97851774e-01 2.71516470e-01 9.00410626e-01
      5.22963654e-01 1.57930660e-02 6.26126273e-01 2.33563869e-01
      7.99212842e-02 3.31294846e-02 6.73446174e-01 8.56326738e-01
      7.17222399e-01 9.82680061e-01 6.99817880e-01 5.80988149e-01
      8.65116696e-01 6.01000908e-01]]
    

There's also the option to set it for all arrays using the below command:


```python
# https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html
np.set_printoptions(threshold = np.inf)
```


```python
x
```




    array([[9.09493927e-01, 2.39865472e-02, 2.55512699e-01, 5.86328195e-01,
            7.11310838e-02, 3.43906434e-01, 1.55961957e-02, 4.27471885e-01,
            7.55280892e-01, 8.91569347e-01, 6.85674566e-01, 3.53757164e-01,
            1.10756416e-01, 2.19419896e-01, 1.48348921e-01, 8.86062128e-01,
            3.29159187e-01, 3.42868380e-02, 7.20196429e-01, 9.40518704e-01,
            1.91526779e-01, 5.78030490e-01, 9.09834624e-01, 3.31523512e-01,
            6.25675801e-01, 8.29793218e-01, 1.98280128e-01, 4.94173211e-01,
            8.84756044e-01, 5.16241643e-01, 4.12229321e-01, 2.69868841e-02,
            2.62166565e-01, 8.43344805e-01, 5.28991601e-01, 3.51315584e-01,
            6.04624653e-01, 6.07264748e-01, 9.48114311e-01, 4.78166438e-01,
            6.68048594e-01, 8.59508647e-01, 2.65764461e-01, 9.68481541e-01,
            1.93221498e-01, 4.45372291e-01, 9.15174093e-01, 2.46202160e-01,
            5.89698895e-01, 3.21986029e-02],
           [8.77345607e-01, 6.58711207e-01, 9.02516252e-01, 9.87827993e-01,
            5.40404639e-01, 4.68443738e-01, 1.78858010e-01, 8.66190209e-01,
            5.84171129e-01, 7.28585681e-01, 5.76629159e-01, 6.28880944e-01,
            8.99590495e-01, 2.58480230e-01, 4.28828180e-01, 3.25574100e-01,
            3.88793491e-01, 9.47620604e-01, 5.96951521e-01, 4.44672279e-01,
            5.51454680e-01, 1.15206288e-01, 1.42543001e-01, 6.21700899e-01,
            9.79686413e-01, 2.67082399e-02, 3.32875674e-01, 7.29076235e-01,
            4.06204298e-04, 4.80769468e-03, 7.68993273e-01, 3.02584403e-01,
            1.48593816e-01, 9.40445524e-01, 1.86897046e-01, 1.21953422e-01,
            6.03991981e-01, 5.03449853e-01, 2.76685087e-01, 3.65547826e-01,
            4.24615207e-01, 7.63027836e-01, 6.68590876e-01, 3.47521237e-01,
            7.83400712e-01, 9.72431596e-02, 7.75169006e-01, 1.08304787e-01,
            1.70414332e-01, 9.35440419e-01],
           [9.00761296e-01, 4.24074718e-01, 4.57967226e-02, 6.69926822e-01,
            8.13062571e-02, 4.24992824e-01, 1.57816435e-01, 4.49856267e-01,
            8.67017093e-01, 5.85894195e-01, 5.58185041e-01, 3.79443337e-01,
            3.06991103e-01, 9.13087437e-01, 8.53450577e-01, 1.14077617e-01,
            2.69261506e-01, 2.44527850e-01, 5.62002921e-01, 3.21048162e-01,
            7.03234477e-01, 6.96583530e-01, 7.00936075e-01, 2.86291409e-01,
            3.47557687e-01, 7.36594848e-01, 1.53182815e-01, 4.08231157e-01,
            4.05792766e-02, 6.13423617e-01, 2.29961660e-01, 9.56343702e-01,
            8.88213692e-01, 3.07395235e-01, 9.69745029e-01, 8.09560031e-01,
            8.68280755e-01, 7.60896385e-01, 4.66355383e-01, 5.35764485e-01,
            4.06991140e-01, 2.40143821e-01, 7.49911318e-01, 6.10924384e-01,
            5.97709273e-01, 7.66935214e-01, 1.68155516e-02, 5.32416390e-01,
            3.97378127e-01, 1.58381915e-01],
           [5.85832676e-01, 4.69261970e-02, 4.49434847e-01, 3.93760601e-01,
            7.89355117e-01, 3.53199694e-01, 5.72347779e-01, 6.01739135e-01,
            1.67563696e-01, 3.99907027e-01, 5.45776104e-01, 6.30217635e-01,
            1.30219306e-01, 9.07390217e-01, 7.91456584e-01, 3.44962215e-01,
            6.70328327e-01, 1.27873375e-02, 1.19891384e-01, 4.08509191e-01,
            9.61505654e-01, 2.06156903e-01, 7.16620516e-01, 7.87112304e-01,
            4.07001026e-01, 8.01709504e-02, 1.35851157e-01, 3.59771384e-01,
            4.19368031e-01, 1.76208130e-01, 5.47808353e-01, 8.94358692e-01,
            9.68440325e-01, 5.96414556e-02, 6.53637163e-01, 8.10309288e-01,
            9.82330863e-01, 7.75340636e-01, 6.87218708e-01, 5.58430316e-01,
            3.65192408e-01, 3.48994339e-01, 6.80945362e-01, 1.93069374e-01,
            1.18215972e-01, 1.08832242e-01, 1.43459333e-01, 5.69379925e-01,
            1.56327091e-01, 1.21767518e-01],
           [2.24502844e-01, 9.48486015e-01, 3.32807718e-01, 9.81136468e-01,
            4.63648396e-01, 8.30050425e-01, 4.72812349e-01, 7.52662178e-01,
            3.69170471e-01, 2.40943428e-01, 9.36711205e-01, 3.55939723e-01,
            4.91873493e-01, 1.42478978e-01, 5.34181068e-01, 1.28885039e-01,
            3.65678160e-01, 7.23522758e-01, 6.75134163e-01, 5.77673421e-01,
            5.44011638e-01, 5.06741244e-01, 4.60582211e-01, 9.49955718e-01,
            1.82737754e-01, 1.88716047e-02, 1.70006544e-01, 2.63572285e-01,
            5.20921547e-01, 3.38468366e-01, 5.44394611e-01, 6.92905927e-01,
            4.58719679e-01, 2.46288435e-01, 8.70814439e-01, 8.87593144e-01,
            6.24917117e-02, 7.18742286e-01, 3.51460287e-02, 9.89762660e-01,
            1.67179022e-01, 9.05753274e-01, 7.47057685e-01, 9.68457248e-01,
            1.80235184e-01, 5.53476121e-01, 1.56274503e-01, 1.53031705e-01,
            7.77363834e-01, 3.57955720e-01],
           [6.44103448e-01, 6.10652955e-01, 8.61528482e-01, 1.86192132e-01,
            7.57056675e-01, 6.25917972e-03, 8.88997645e-01, 5.60190295e-01,
            7.78102085e-01, 7.68879259e-01, 5.08866976e-01, 4.14625899e-01,
            7.45531934e-01, 9.42978416e-01, 5.15441545e-02, 5.07498738e-01,
            6.22196385e-01, 1.31233597e-01, 1.77969975e-01, 1.76071555e-04,
            3.05662794e-01, 7.32209704e-01, 8.23953013e-01, 4.67618791e-01,
            5.97785255e-01, 5.16311088e-01, 7.33762585e-02, 1.01120547e-01,
            2.68163995e-01, 5.35862790e-01, 7.80980015e-01, 6.01414149e-01,
            5.43838023e-02, 8.22605155e-02, 2.86067059e-01, 3.22367166e-01,
            2.34773916e-01, 5.81265042e-01, 2.83901549e-01, 8.08501149e-01,
            2.50157001e-01, 3.98508576e-02, 4.29980951e-01, 1.94903927e-01,
            8.65610184e-01, 7.02643108e-01, 9.30058975e-02, 5.80190079e-01,
            6.64651633e-01, 5.83821269e-01],
           [2.26297379e-01, 1.90022710e-01, 9.53720667e-01, 5.14270734e-01,
            4.15346281e-01, 4.96314544e-01, 4.52366417e-01, 5.94662560e-01,
            7.91297007e-01, 2.07625687e-01, 5.63166422e-01, 6.56349961e-01,
            4.71712611e-01, 2.75534331e-01, 8.49809005e-01, 1.56118581e-01,
            2.13215209e-01, 3.06627861e-01, 3.71058907e-01, 5.62302991e-01,
            3.17328855e-01, 8.73208874e-01, 4.09203959e-01, 5.71581260e-01,
            3.17169135e-01, 4.05446568e-01, 7.96942267e-01, 1.56060445e-01,
            4.42942172e-01, 8.01147077e-01, 2.59941549e-03, 8.08003680e-01,
            4.39053569e-01, 8.64538148e-01, 9.08317809e-01, 5.42054866e-01,
            4.30537684e-01, 8.00627252e-01, 7.02198618e-01, 9.07857650e-01,
            9.64071352e-01, 4.16449229e-01, 2.17517449e-01, 4.80606284e-01,
            7.12293354e-01, 1.06732510e-02, 3.10536380e-01, 9.22633818e-01,
            6.81666189e-01, 6.24167804e-01],
           [6.79900529e-01, 1.75375765e-01, 7.74831103e-01, 4.51360411e-01,
            7.84686889e-01, 4.83854048e-01, 4.53583053e-01, 8.99983658e-01,
            2.90939227e-02, 2.03570207e-01, 7.20679809e-01, 9.34854654e-01,
            8.99149364e-01, 5.63587400e-02, 5.63211048e-01, 4.87033109e-01,
            2.75330136e-01, 8.95884694e-01, 5.37172601e-01, 8.80532007e-01,
            1.41171486e-01, 8.01480329e-01, 9.00111518e-01, 8.74942736e-01,
            8.49791237e-01, 3.99779403e-01, 8.44924218e-01, 1.22134667e-01,
            6.65183025e-01, 7.30792079e-01, 9.24981066e-01, 5.96340101e-01,
            1.71169951e-01, 7.81357782e-01, 2.59912011e-01, 3.27655470e-01,
            6.74396196e-01, 4.09835467e-01, 6.23541561e-01, 4.34381794e-01,
            1.81901357e-01, 6.96795979e-01, 9.98273369e-02, 3.90759757e-01,
            4.02720018e-01, 9.43538466e-01, 9.40825161e-01, 2.56271182e-01,
            2.92165169e-01, 4.26202172e-01],
           [2.69545542e-01, 5.61694184e-01, 8.03679368e-01, 1.64893844e-01,
            3.95601910e-01, 2.91093083e-01, 5.94307198e-01, 1.22809162e-01,
            6.00669507e-01, 9.46186358e-01, 4.89280074e-02, 8.65981275e-01,
            6.11723212e-01, 2.43525055e-01, 7.93425197e-01, 8.79930422e-01,
            9.30703814e-01, 3.95002001e-01, 2.80047450e-01, 5.59755256e-01,
            7.19201792e-01, 3.20396194e-01, 8.10228635e-02, 2.77217469e-01,
            6.98872573e-01, 2.34750614e-01, 2.76818363e-01, 2.43105286e-02,
            3.98969179e-01, 5.85609177e-01, 4.99157641e-02, 3.78621719e-01,
            2.11882347e-01, 1.02583768e-01, 7.82066037e-01, 3.53437154e-01,
            7.06462578e-01, 4.41651121e-01, 9.32809328e-01, 9.50072870e-01,
            4.55079788e-01, 3.10588072e-02, 2.58934505e-01, 2.18415285e-01,
            2.81612870e-02, 7.35993055e-01, 7.92541552e-01, 9.59728281e-01,
            1.98256211e-01, 7.02334701e-01],
           [1.61601982e-01, 8.22810875e-01, 8.57382420e-01, 8.50873386e-01,
            1.75159180e-01, 9.93365268e-01, 4.58487116e-01, 8.56046915e-01,
            2.09695512e-01, 2.06523355e-01, 4.45189109e-01, 4.63978499e-01,
            6.78257832e-01, 3.60323355e-02, 1.83188961e-02, 2.13447350e-01,
            4.70807303e-01, 5.86784970e-01, 7.49137860e-01, 7.98306992e-01,
            7.44758307e-01, 9.60715199e-01, 9.09112152e-01, 2.35451493e-01,
            2.95013959e-01, 9.07803306e-01, 4.43469907e-01, 5.96816275e-01,
            9.08894465e-01, 2.26281771e-01, 7.91623177e-01, 6.31917572e-01,
            4.46217661e-01, 8.21809641e-01, 8.38562071e-01, 4.53824709e-01,
            9.92758966e-01, 7.32596228e-01, 3.26920666e-01, 7.36156045e-01,
            1.12227943e-01, 6.61665942e-01, 1.71657664e-01, 6.45049338e-01,
            8.64207521e-01, 7.88614831e-01, 9.55863205e-01, 2.17488735e-01,
            1.87450463e-01, 9.99267725e-01],
           [6.77134178e-02, 7.98943004e-01, 7.60731941e-02, 4.31534582e-02,
            5.04917078e-01, 7.91571716e-01, 6.64867343e-01, 5.60482103e-01,
            5.91621164e-01, 5.10131195e-01, 3.06537241e-01, 8.41231744e-01,
            9.74857525e-01, 8.89182445e-02, 5.00240548e-02, 2.60717570e-01,
            8.53998697e-01, 4.61732505e-01, 6.32170213e-01, 5.55705402e-01,
            3.55734775e-01, 8.03635509e-01, 7.32915671e-01, 6.46466484e-01,
            3.76342040e-01, 5.28209875e-01, 5.57619492e-01, 2.94673140e-01,
            7.00268348e-01, 4.62178177e-01, 9.39428117e-01, 7.63576114e-01,
            1.73729812e-01, 1.88256063e-01, 8.53717168e-02, 2.21594964e-01,
            1.83469270e-01, 7.26077403e-01, 3.04076658e-01, 6.95961183e-02,
            7.00311650e-01, 8.28326598e-01, 1.42210571e-01, 2.66220893e-01,
            4.71491960e-01, 1.71255982e-01, 7.60325963e-01, 7.99322264e-01,
            8.98345823e-01, 4.69914376e-03],
           [1.60392551e-01, 5.41962693e-01, 5.80370963e-01, 6.33619092e-01,
            7.96169087e-01, 4.19931270e-01, 3.49820045e-01, 8.58232840e-01,
            6.27988163e-01, 7.82169975e-01, 9.02547620e-01, 7.05029466e-02,
            4.07994048e-01, 4.92399171e-01, 1.61680470e-01, 4.79936532e-01,
            3.15943925e-01, 7.48017998e-01, 4.43633150e-02, 6.60536663e-01,
            7.78206940e-02, 5.89816935e-01, 2.06733124e-02, 5.65564014e-01,
            9.99321781e-01, 6.59645779e-01, 7.63963049e-01, 8.40377792e-01,
            6.15912592e-02, 5.93778956e-01, 8.74509137e-01, 1.82305575e-01,
            8.95003312e-01, 2.33764797e-01, 7.04413553e-01, 3.45911600e-01,
            9.77750522e-01, 6.14799922e-01, 8.07888004e-01, 5.19101916e-01,
            3.19441432e-01, 5.93872905e-01, 1.75834975e-01, 2.24650235e-01,
            2.01086076e-01, 3.80821750e-01, 7.30454687e-02, 1.41431801e-01,
            6.69907037e-01, 7.40191359e-02],
           [3.30593859e-01, 7.40289231e-01, 7.98166307e-01, 7.83321444e-01,
            8.65575729e-01, 2.98719822e-01, 1.58527585e-01, 7.08109818e-01,
            6.80408612e-01, 7.35117242e-01, 9.43761181e-01, 7.67959137e-01,
            9.97013241e-01, 3.67101011e-01, 7.63258600e-01, 7.45133816e-01,
            5.30847187e-01, 4.74846305e-01, 4.06200673e-01, 7.80749890e-01,
            6.63707802e-01, 2.60748212e-01, 8.67788019e-01, 8.54440487e-01,
            1.44999959e-02, 7.09314964e-01, 1.88188663e-01, 2.58823709e-01,
            7.95493701e-01, 8.26785449e-01, 2.77295828e-01, 2.86020608e-01,
            4.12981863e-01, 2.07092008e-02, 7.68729594e-01, 9.45194355e-01,
            7.38292744e-03, 4.23737220e-01, 6.31703107e-01, 4.46821555e-01,
            3.62170232e-01, 1.91218924e-01, 1.22643423e-01, 8.21479234e-01,
            9.02215499e-01, 1.84815071e-01, 4.01174243e-01, 3.14879922e-01,
            3.77497100e-01, 7.44965562e-01],
           [7.77053921e-01, 2.35419489e-01, 7.09578833e-01, 2.21537035e-01,
            5.01335063e-01, 7.11603835e-01, 5.94046607e-01, 4.64221570e-01,
            8.17429579e-01, 3.14442202e-01, 9.27260889e-01, 3.20730930e-01,
            2.75314918e-01, 1.85956643e-01, 2.58579123e-01, 9.22509013e-01,
            9.14877648e-01, 4.40313934e-01, 5.87450265e-01, 5.94797055e-01,
            4.13738355e-01, 1.98652500e-01, 4.41664266e-01, 2.88317727e-02,
            7.91974361e-01, 4.29381240e-01, 8.64920703e-01, 2.88044648e-01,
            6.33779159e-01, 8.84159319e-01, 5.99882141e-01, 6.53870277e-01,
            8.37018884e-01, 1.36643176e-02, 4.74381645e-01, 8.03322656e-01,
            5.39102576e-01, 2.24950397e-01, 2.52444626e-01, 3.50411071e-01,
            6.78879308e-01, 2.56271077e-01, 6.51230836e-01, 5.57034577e-02,
            3.48867799e-01, 8.33588663e-01, 6.38824005e-01, 7.99137490e-01,
            4.37760196e-01, 3.55883574e-01],
           [6.84361830e-01, 3.16652828e-01, 1.97257899e-01, 1.26678784e-01,
            7.60471767e-02, 3.99468039e-02, 6.23442327e-01, 8.99956097e-01,
            2.91457866e-01, 5.76907513e-01, 7.60338575e-01, 7.31439993e-01,
            1.15080243e-01, 8.88226949e-01, 6.87801993e-01, 4.01423651e-01,
            2.73495184e-01, 4.71115420e-01, 4.06356182e-01, 7.89409613e-01,
            1.23422717e-01, 4.66277057e-02, 5.98017748e-01, 2.95715418e-01,
            1.31222257e-01, 8.33206169e-01, 8.70153102e-01, 8.39841716e-01,
            8.90747661e-01, 6.43582815e-01, 9.26239058e-01, 2.58899402e-01,
            4.60422794e-01, 8.80037888e-01, 2.53905685e-01, 6.17429522e-01,
            3.13637282e-01, 9.77377841e-03, 3.97574982e-02, 6.21990477e-01,
            8.91024606e-01, 2.87192945e-01, 6.01163842e-01, 7.90091509e-01,
            3.62065478e-01, 9.00887675e-01, 3.51150935e-01, 7.25035058e-01,
            5.47455992e-01, 8.93993085e-02],
           [1.41617724e-01, 3.26927062e-01, 7.37450230e-01, 8.99632920e-01,
            5.17782844e-01, 5.48095240e-01, 5.76335935e-02, 1.47010595e-01,
            5.31332433e-01, 9.05799929e-01, 5.20380505e-01, 3.14046879e-01,
            5.45301668e-01, 2.79731045e-01, 3.58299592e-01, 8.79768424e-01,
            3.77795667e-01, 8.43499989e-01, 1.61886591e-01, 7.41574283e-01,
            9.99107495e-02, 9.96665026e-01, 9.86857781e-01, 3.50404680e-01,
            5.87686635e-01, 5.88893606e-01, 5.24143412e-01, 7.57274731e-01,
            6.41220915e-01, 5.35189031e-01, 9.47234170e-01, 5.60587349e-01,
            4.37504001e-01, 1.98195787e-01, 2.21247270e-01, 4.61333812e-01,
            1.63166363e-01, 1.46015916e-01, 6.87412450e-01, 2.42748813e-02,
            2.81201528e-01, 8.60424795e-01, 7.51798637e-01, 6.86093905e-01,
            3.59895278e-01, 7.71966856e-03, 8.78946826e-01, 3.96973299e-02,
            1.34644385e-01, 1.48922696e-01],
           [4.83211612e-01, 2.11342776e-01, 3.89103437e-01, 4.88028332e-01,
            6.33244344e-01, 3.19890221e-01, 8.86224029e-01, 7.81489121e-01,
            7.16106184e-01, 9.47798828e-01, 7.61932851e-01, 8.89782948e-01,
            1.52267166e-01, 6.06834899e-02, 1.32222714e-01, 6.34730241e-01,
            6.53190612e-01, 6.31511469e-02, 5.90230543e-01, 7.13097156e-01,
            2.84799306e-01, 4.08583489e-01, 3.88959627e-01, 2.49390621e-01,
            6.88725139e-02, 6.93363158e-01, 4.35914765e-01, 7.22825493e-01,
            6.12449813e-01, 8.67747596e-01, 5.80395211e-01, 3.71339637e-01,
            1.01786572e-01, 5.73240918e-01, 7.72163484e-01, 5.21016457e-01,
            3.22519750e-01, 6.29818015e-01, 6.10760201e-01, 4.92553435e-01,
            2.59542722e-01, 9.16245923e-01, 4.22026093e-01, 6.77235848e-01,
            1.91199895e-01, 2.84449660e-01, 4.34501782e-01, 2.30775641e-01,
            5.56471459e-01, 2.03636233e-01],
           [5.28042313e-01, 4.11320878e-01, 5.43472397e-01, 7.82162323e-01,
            4.01639241e-01, 8.52040787e-02, 5.81047802e-01, 4.90939472e-01,
            5.97261127e-01, 3.51904431e-01, 3.88029398e-01, 5.42341471e-01,
            8.31136123e-01, 6.98232876e-01, 4.24420813e-01, 5.55977688e-01,
            2.42794927e-01, 7.98560723e-01, 2.19193906e-02, 6.68728955e-01,
            8.92382350e-01, 2.25810722e-01, 4.32944238e-01, 9.77840962e-01,
            5.84776466e-01, 3.47453813e-01, 3.56642581e-01, 5.21073768e-01,
            2.99407001e-01, 5.68581105e-02, 1.28260341e-02, 7.82262515e-01,
            2.89734652e-02, 2.88041194e-01, 4.19198347e-01, 7.62481881e-01,
            5.92376405e-01, 7.26227339e-01, 7.86324893e-01, 2.09136571e-02,
            3.52724702e-01, 7.34158623e-01, 2.53742494e-01, 6.25225634e-01,
            5.11872721e-01, 2.27319814e-01, 1.93031813e-01, 4.44715423e-01,
            1.26344445e-01, 6.18300856e-01],
           [3.36575372e-02, 3.28284806e-01, 1.74337979e-01, 1.96717856e-01,
            6.06402568e-01, 1.99231570e-02, 9.46651758e-01, 7.90928566e-01,
            4.91134933e-01, 7.26075816e-01, 8.49137811e-02, 2.11978397e-01,
            1.95695004e-01, 3.58949207e-01, 1.46670137e-01, 4.55862629e-01,
            9.90806079e-01, 7.25271839e-01, 7.13620755e-01, 5.99640169e-01,
            3.04706807e-01, 9.58247522e-01, 1.62534947e-01, 8.72165707e-01,
            1.85745176e-01, 1.62066991e-02, 6.61045176e-01, 9.23840001e-01,
            7.17915734e-01, 4.36930158e-01, 5.61376219e-01, 6.98545993e-01,
            9.16597884e-01, 4.02984117e-01, 9.13172538e-01, 7.81888460e-01,
            3.99686245e-01, 2.08217306e-01, 7.49743174e-01, 8.88932245e-01,
            5.17586965e-01, 6.42383421e-01, 4.34892984e-01, 1.12229419e-01,
            7.10137390e-01, 9.20241124e-01, 5.25063245e-02, 2.14008777e-01,
            1.45656974e-01, 5.89479743e-01],
           [6.60335812e-01, 5.90024828e-01, 6.49940848e-01, 7.21829904e-01,
            1.37624774e-01, 8.27112220e-01, 8.90654973e-03, 6.78435553e-01,
            7.13581431e-01, 7.73961495e-01, 7.86381640e-01, 1.40685484e-01,
            6.83692065e-02, 7.40419628e-01, 9.42222499e-01, 2.29840685e-01,
            8.83614300e-01, 8.54835525e-01, 2.73841666e-01, 1.29301345e-01,
            8.97160103e-01, 4.04563706e-01, 1.07817499e-01, 9.83734260e-01,
            1.81653598e-01, 6.35045672e-01, 6.14803012e-01, 6.27673659e-01,
            6.78672764e-01, 1.03498762e-01, 4.16057356e-02, 9.49273813e-01,
            7.41503765e-01, 2.20029735e-01, 3.86060612e-01, 7.84661497e-01,
            3.69850254e-02, 9.97613212e-01, 5.87726663e-01, 2.88472957e-02,
            9.80929305e-01, 1.44050606e-01, 6.51122301e-01, 5.19613285e-01,
            7.15414651e-01, 7.48279013e-01, 3.50958714e-01, 5.16836088e-01,
            1.71377722e-01, 1.61327213e-01],
           [4.12075595e-01, 2.17900699e-01, 3.55076707e-01, 9.59830519e-01,
            5.47232274e-02, 2.23339089e-01, 5.22411055e-01, 9.65150110e-01,
            7.41732391e-01, 3.04323043e-01, 8.93653400e-01, 2.37356917e-01,
            8.38045436e-01, 5.89849595e-01, 7.46249821e-01, 1.70236936e-01,
            2.81571761e-01, 9.95551894e-02, 8.00219149e-01, 7.31865913e-01,
            9.58400083e-01, 3.80146512e-01, 3.80116585e-01, 9.62082322e-01,
            6.32716028e-01, 8.84041340e-01, 1.42174107e-01, 4.08699286e-01,
            4.82625234e-01, 3.11603529e-01, 2.10748967e-01, 5.26729960e-01,
            1.70571041e-01, 7.88818064e-01, 1.30365418e-01, 1.36499916e-01,
            1.30308219e-01, 4.26193251e-01, 1.47779766e-01, 5.57586180e-01,
            1.62465069e-01, 5.10190670e-02, 1.13472925e-01, 5.25956737e-01,
            1.97359722e-01, 3.98233236e-01, 7.52937170e-01, 9.42408671e-01,
            2.39461753e-01, 8.10087155e-01],
           [8.98743796e-02, 3.93319418e-01, 8.67469763e-01, 8.47044040e-01,
            8.47435319e-01, 1.17176908e-01, 3.36886972e-01, 8.41152881e-01,
            3.11858411e-01, 2.34784646e-01, 4.24213940e-01, 4.12674845e-01,
            3.73540059e-01, 6.62450157e-01, 3.65885620e-01, 7.41667896e-01,
            7.11859435e-02, 3.19276686e-01, 9.90213054e-01, 8.46348955e-01,
            7.07469654e-01, 5.47804787e-01, 8.45357140e-01, 9.19110201e-01,
            3.91634374e-01, 9.82410034e-01, 7.44038954e-01, 8.27533975e-01,
            3.56214080e-01, 7.76939575e-01, 4.86318543e-01, 3.26370047e-01,
            4.31871367e-01, 7.44016852e-01, 4.16789099e-01, 4.09098266e-01,
            7.46575360e-01, 5.62792261e-02, 6.57659504e-02, 7.88813467e-01,
            1.54101913e-01, 3.97015843e-01, 8.93668828e-01, 7.10012471e-01,
            3.58170535e-01, 4.53257981e-01, 1.32801695e-01, 1.53608207e-01,
            6.43725286e-01, 2.39397302e-01],
           [5.92478371e-01, 7.10386573e-02, 6.43824346e-01, 4.37153586e-01,
            5.50041803e-01, 7.32301382e-01, 7.80527238e-01, 3.12694816e-01,
            2.48292673e-01, 5.99210765e-01, 5.43997835e-01, 4.24233870e-01,
            1.66778828e-01, 5.13229559e-01, 8.07090311e-01, 2.74487079e-01,
            5.34546731e-01, 8.29036687e-01, 6.86931585e-01, 2.93729122e-01,
            7.06311706e-01, 6.57466558e-01, 6.13369677e-01, 8.61775390e-01,
            2.40748354e-01, 3.39624568e-01, 2.78622635e-01, 7.17970414e-02,
            9.90659117e-01, 2.81590797e-01, 3.66397331e-01, 2.14078646e-01,
            7.42824051e-01, 1.35995891e-01, 4.19733608e-02, 8.79760595e-01,
            1.67987788e-01, 3.54636797e-01, 3.10008705e-01, 3.62871222e-01,
            5.04966029e-01, 4.84380573e-01, 7.95457248e-02, 4.20032554e-01,
            2.13480489e-02, 5.37209984e-02, 1.32871020e-01, 1.92958663e-01,
            9.54114448e-01, 4.04372340e-01],
           [8.05879745e-01, 9.02487545e-01, 5.21292164e-01, 1.22248519e-01,
            3.93834242e-01, 8.97451946e-01, 1.26920820e-02, 9.63549754e-01,
            6.48947841e-01, 5.82222807e-01, 6.46400557e-01, 7.78306066e-01,
            6.11904887e-01, 1.34928446e-01, 9.14806380e-01, 3.11983886e-01,
            9.68564057e-01, 2.24152252e-01, 8.30691840e-01, 3.68626664e-01,
            9.20400704e-01, 1.89308914e-01, 4.32956656e-01, 9.90321701e-01,
            3.58273162e-01, 7.10955101e-01, 2.69503379e-01, 9.37914984e-01,
            5.93787966e-01, 4.30482244e-01, 9.66779309e-01, 4.62536889e-01,
            8.28494133e-01, 3.60502875e-01, 9.30781040e-01, 5.27588299e-01,
            3.81180689e-01, 7.71739635e-01, 3.71208160e-01, 1.55923174e-01,
            4.90518222e-01, 7.80761612e-01, 4.84916383e-01, 8.72735794e-01,
            2.35262208e-01, 2.39182303e-02, 1.70430238e-01, 2.58213049e-01,
            7.42570981e-01, 8.57315083e-01],
           [3.78637004e-01, 7.06029925e-01, 3.67714518e-02, 2.91109303e-01,
            8.25940750e-01, 2.96454524e-01, 7.40251167e-01, 9.75473325e-01,
            6.01355530e-01, 9.59431582e-01, 5.08018680e-01, 7.45247439e-01,
            1.48648806e-01, 5.26337365e-02, 8.62423675e-01, 9.12985188e-01,
            4.16586737e-01, 1.55633370e-01, 8.12244714e-01, 4.94405258e-01,
            5.88303573e-01, 5.84922648e-01, 5.73994204e-01, 3.16627542e-01,
            1.64149496e-02, 4.77506196e-01, 9.53456486e-01, 1.83976006e-01,
            5.99057154e-01, 1.27598759e-01, 8.54932136e-01, 1.33327996e-01,
            7.25956077e-01, 2.67633363e-01, 1.85201179e-01, 1.52416410e-01,
            5.23768657e-01, 5.18796429e-01, 2.51048248e-01, 7.45534081e-01,
            7.79121647e-01, 5.69040139e-01, 7.68762413e-01, 6.21715699e-01,
            4.67321421e-01, 7.82091490e-01, 9.18734500e-01, 3.21732444e-01,
            7.70242133e-01, 5.61629744e-01],
           [3.26231074e-01, 2.86861910e-01, 5.68731038e-01, 5.23030448e-01,
            6.93229991e-01, 1.86302847e-01, 1.02598615e-01, 7.82906691e-01,
            2.25924779e-01, 6.95240972e-01, 2.28906149e-01, 1.71802263e-01,
            2.00421378e-01, 7.05552421e-01, 8.67238656e-01, 5.74599352e-01,
            2.88871884e-01, 4.75244802e-01, 5.65055068e-01, 4.72919972e-01,
            5.68022017e-01, 3.10222669e-01, 6.07368176e-01, 9.97920264e-01,
            2.12876986e-02, 7.96269926e-01, 3.06637682e-01, 5.44683767e-01,
            8.13148721e-01, 6.02823428e-01, 9.83427058e-01, 9.29673840e-01,
            1.95066438e-01, 6.20013585e-01, 4.22968015e-01, 6.79969793e-01,
            2.45124465e-01, 8.35467967e-01, 5.04620537e-01, 5.48662464e-01,
            1.29728716e-01, 8.07737242e-01, 3.07115715e-01, 2.78329013e-01,
            5.23942105e-01, 1.29304757e-01, 1.86638057e-01, 8.69679338e-01,
            7.98436451e-01, 6.55591732e-01],
           [4.82511809e-01, 2.66054600e-01, 2.42780615e-02, 3.53509827e-01,
            7.97089900e-03, 2.36605642e-01, 2.65821101e-01, 5.97227352e-01,
            7.84152372e-01, 9.17176006e-01, 6.96059567e-01, 5.61981345e-01,
            2.77069620e-01, 4.58277801e-01, 5.42093325e-02, 9.11540275e-01,
            9.82658680e-02, 2.58616683e-01, 1.15678338e-01, 6.44295110e-01,
            1.41292658e-01, 9.55339681e-01, 9.18835649e-01, 1.80581786e-01,
            6.54665249e-01, 4.21558516e-01, 8.27160856e-01, 6.16151805e-01,
            1.80496016e-01, 7.96135434e-01, 7.92771500e-01, 8.39724201e-01,
            2.26053672e-01, 7.18529337e-01, 8.01843393e-01, 3.27201526e-01,
            3.42405196e-03, 2.64285076e-01, 4.18077968e-01, 5.58529030e-01,
            1.20624091e-01, 1.88947971e-01, 9.51877899e-01, 9.14027997e-01,
            2.60244873e-01, 5.87403742e-01, 6.73820421e-01, 7.67253443e-01,
            2.01166371e-01, 1.34691702e-01],
           [6.49530972e-01, 5.02199364e-02, 2.53575261e-01, 2.74647194e-01,
            1.58952376e-01, 7.24724109e-01, 8.95431085e-01, 6.87151658e-01,
            8.63698938e-01, 6.78888018e-01, 8.64306578e-01, 4.14001212e-02,
            7.71836190e-01, 5.06367961e-01, 5.08737247e-01, 9.04501193e-01,
            1.76456317e-01, 1.18610668e-02, 1.63570143e-01, 1.90361250e-01,
            1.23885101e-01, 3.48241664e-01, 3.77872706e-01, 6.52316468e-01,
            6.32991823e-01, 3.51716520e-01, 7.15706253e-01, 9.89268377e-01,
            6.62415154e-01, 5.73149141e-01, 7.70069694e-01, 5.44650782e-01,
            5.13901352e-01, 7.70927542e-01, 8.19342672e-01, 9.22794671e-01,
            5.69373660e-02, 2.27219361e-01, 3.79948534e-01, 6.82911579e-01,
            1.21980135e-01, 5.78915187e-01, 9.97900298e-02, 3.23915824e-02,
            5.50844764e-03, 8.72689839e-01, 3.29136229e-01, 7.92236161e-02,
            9.38884842e-01, 7.38260139e-01],
           [4.94360428e-01, 1.03558746e-01, 3.17316527e-02, 3.05614713e-01,
            2.66055152e-01, 7.05146366e-01, 4.55481800e-01, 9.66495342e-01,
            7.93175837e-01, 7.37773406e-02, 2.72047360e-01, 7.87452036e-01,
            5.21251506e-01, 2.94650040e-01, 2.82343610e-01, 8.13731391e-01,
            5.53945065e-01, 4.23775249e-01, 3.51794516e-01, 7.20660209e-01,
            7.74859009e-02, 9.31206333e-01, 3.97680817e-01, 6.51535472e-01,
            4.19334384e-01, 3.65948638e-01, 7.31140806e-01, 3.43629156e-01,
            9.00930793e-02, 5.80093651e-02, 5.70139457e-01, 9.02487063e-01,
            8.90036056e-03, 1.17754995e-01, 2.98489693e-01, 3.14390103e-02,
            7.94638114e-01, 7.30042223e-01, 3.70064123e-01, 7.86345453e-01,
            4.85150573e-01, 7.18064393e-01, 2.17741755e-01, 1.40080478e-03,
            9.31104308e-01, 4.98848708e-01, 4.78951322e-02, 6.33677802e-01,
            3.50582735e-01, 7.58812339e-01],
           [6.64298180e-01, 4.48134136e-01, 5.40221728e-02, 4.15713827e-03,
            4.73529144e-02, 3.24834272e-01, 1.48990508e-01, 5.98453323e-01,
            3.51049310e-01, 1.47205752e-01, 5.87700912e-01, 7.29046135e-01,
            4.68816832e-01, 8.18945463e-02, 1.26726252e-01, 5.74167470e-01,
            9.40781132e-01, 4.49903491e-01, 2.24380925e-01, 3.10749826e-01,
            8.04094268e-01, 3.09511112e-01, 4.57434275e-01, 3.75617563e-01,
            5.71064562e-02, 5.50079558e-01, 4.44212104e-01, 7.77155064e-01,
            2.72738179e-01, 3.50739275e-01, 6.50776902e-01, 8.87102060e-01,
            3.93688738e-01, 7.46466066e-01, 9.27755859e-01, 5.22213051e-01,
            3.71458513e-01, 5.87165920e-01, 6.58633973e-01, 8.65281658e-01,
            8.89576662e-01, 3.23110054e-01, 3.64237399e-01, 2.69940448e-01,
            5.79820990e-01, 4.75327762e-01, 8.94432652e-01, 7.78382183e-02,
            6.94581555e-01, 4.83675577e-01],
           [5.32973180e-01, 8.92026344e-01, 5.05390928e-02, 1.68257752e-01,
            8.93138785e-01, 9.21862785e-01, 9.59631844e-01, 7.80442489e-01,
            4.75536322e-01, 1.28115503e-01, 9.77614409e-01, 8.07821346e-01,
            1.79455263e-01, 2.49231464e-01, 7.30633421e-01, 6.49643453e-01,
            5.28831188e-02, 6.56147442e-01, 7.74642467e-01, 6.25918669e-01,
            1.36226100e-01, 6.02651503e-01, 5.88287596e-01, 7.14673404e-01,
            1.87829972e-01, 7.53472814e-01, 3.12259089e-01, 7.66906837e-01,
            9.24047973e-01, 1.12139627e-01, 7.34354219e-01, 8.02892261e-01,
            9.63782968e-02, 2.71216198e-01, 2.89517519e-01, 8.52599127e-01,
            1.62309792e-01, 2.88365281e-01, 7.86824469e-01, 6.05109914e-01,
            4.58678077e-01, 6.42330101e-01, 5.47335284e-01, 3.14345921e-01,
            1.65936483e-01, 3.01871238e-01, 8.44254222e-01, 6.41537396e-01,
            1.66742060e-01, 7.54753939e-01],
           [8.19768572e-01, 1.24943677e-01, 2.33175491e-01, 1.91034915e-02,
            1.33178916e-01, 2.52742429e-01, 5.12430412e-01, 3.10073922e-01,
            8.36689379e-01, 6.15859211e-01, 4.79470686e-01, 2.87390564e-01,
            4.55271037e-02, 9.21188119e-01, 3.55621070e-01, 8.81364052e-01,
            5.19543280e-01, 2.91936517e-01, 6.31556934e-01, 4.22156580e-01,
            8.33871994e-01, 1.87318396e-01, 7.48329444e-01, 4.16517801e-03,
            4.34322368e-02, 9.73960201e-01, 6.08745251e-01, 1.62335127e-01,
            9.09819173e-02, 1.03452739e-01, 8.52554333e-01, 7.94041491e-02,
            2.44723333e-01, 1.94306582e-01, 7.09629356e-01, 5.67432247e-01,
            3.99027958e-01, 3.62690709e-01, 7.93570915e-01, 2.28403051e-01,
            4.42390752e-01, 3.52210599e-01, 9.57329078e-01, 1.36428313e-01,
            2.86334427e-02, 5.79196322e-01, 6.16004787e-01, 1.79844125e-01,
            7.97899272e-02, 8.89127390e-01],
           [9.48870138e-01, 1.76315058e-01, 1.46754297e-01, 7.36010899e-01,
            6.21041269e-01, 1.07500960e-01, 2.20622360e-01, 6.07546784e-01,
            7.35378754e-01, 7.80417336e-01, 5.00510630e-01, 7.99200179e-01,
            5.27987010e-01, 9.62864967e-01, 5.43792860e-01, 8.56957927e-01,
            9.82819439e-01, 1.83754925e-01, 3.72543131e-01, 4.56799610e-01,
            2.41910631e-01, 4.75922252e-02, 8.83353524e-01, 6.42575544e-01,
            4.49106621e-01, 6.17295538e-01, 6.03524935e-01, 8.13733762e-01,
            2.47428830e-01, 9.54121051e-01, 7.07907705e-01, 6.82999229e-01,
            3.68114641e-01, 2.57441549e-01, 7.04024133e-01, 9.00871569e-01,
            1.67427942e-01, 6.78761324e-01, 2.09743074e-01, 5.81888362e-01,
            5.80884146e-01, 4.07756745e-01, 8.93696322e-01, 8.14044484e-01,
            1.06057808e-01, 5.64546123e-01, 9.06198657e-01, 3.91225943e-01,
            8.01672827e-01, 2.38335075e-01],
           [8.59950980e-01, 1.85091534e-01, 2.82288353e-01, 2.46907229e-01,
            7.27067686e-01, 2.11703243e-02, 1.10021325e-03, 5.84172702e-01,
            8.75798660e-01, 1.06481615e-01, 9.14070810e-02, 2.09663148e-01,
            9.48718962e-01, 8.13095052e-01, 7.26726469e-01, 8.97068581e-01,
            3.83157624e-01, 4.81488043e-01, 7.68767675e-01, 9.91773247e-01,
            2.35335829e-01, 8.91896723e-01, 6.16467244e-01, 4.09699981e-01,
            3.82360927e-01, 4.48453799e-01, 2.10999381e-01, 8.86483065e-01,
            7.67044279e-01, 7.31271321e-01, 3.16819439e-01, 1.82101537e-01,
            7.14779065e-01, 4.28881562e-01, 5.11690483e-01, 8.00094793e-03,
            6.78340634e-01, 5.78531504e-01, 3.14011038e-01, 4.41427767e-02,
            5.66721437e-01, 9.53193650e-01, 7.32164846e-01, 6.49544208e-02,
            8.50542953e-01, 9.62424273e-01, 5.02664136e-01, 5.51869993e-01,
            7.39756995e-01, 6.73593132e-01],
           [2.21468371e-01, 6.82529696e-01, 1.54419941e-01, 7.47373026e-01,
            4.26441027e-01, 2.25843555e-01, 5.19570558e-01, 9.77858836e-01,
            2.75422599e-01, 2.47171738e-01, 8.81533096e-01, 4.41846451e-01,
            2.38266420e-01, 6.64592957e-01, 5.86150176e-01, 8.34738962e-01,
            6.86837733e-01, 8.30071568e-01, 7.54083032e-01, 9.47480128e-02,
            3.14677012e-01, 4.12037967e-02, 4.19330611e-01, 1.05115647e-01,
            5.10466490e-01, 9.56043361e-02, 6.03028321e-01, 7.09911106e-02,
            6.99281277e-01, 8.10313539e-01, 9.60674257e-01, 6.39095295e-01,
            9.40594104e-01, 4.01508589e-01, 9.80497538e-01, 4.46246222e-01,
            4.16778301e-01, 4.76258657e-01, 4.01213191e-02, 2.00378857e-01,
            3.14792794e-01, 3.95707525e-01, 2.76452600e-01, 5.06984216e-01,
            5.41419890e-01, 2.25459226e-01, 7.14342214e-01, 4.65397314e-01,
            4.54868978e-01, 4.83640337e-01],
           [2.11723113e-01, 1.18020937e-01, 5.74394843e-01, 8.69534374e-01,
            4.82339342e-01, 8.43486876e-01, 9.09842802e-01, 5.32612353e-01,
            2.56363340e-01, 7.92744118e-01, 9.01600159e-01, 1.80116772e-01,
            5.35315472e-01, 2.67447427e-01, 2.46667470e-01, 6.69541409e-01,
            6.51101187e-02, 4.90682956e-01, 7.15824671e-01, 3.58057461e-01,
            5.55041258e-01, 8.81707621e-01, 4.77350057e-02, 8.26420660e-01,
            6.24315104e-01, 3.57662328e-01, 8.78390991e-01, 5.24480601e-01,
            6.83730700e-01, 1.14068528e-01, 4.40325932e-01, 5.87051129e-01,
            6.52122481e-01, 8.63816990e-01, 3.22548122e-01, 3.97400424e-01,
            6.03810585e-01, 1.18216101e-01, 2.13552987e-01, 7.50158531e-01,
            5.22924822e-01, 4.36510532e-02, 8.32066793e-01, 4.45261122e-01,
            7.06814188e-01, 8.61820929e-02, 7.82866668e-01, 8.67835431e-02,
            6.83776476e-01, 8.54748189e-01],
           [7.28414023e-01, 6.28000219e-01, 6.07992183e-01, 7.95909939e-01,
            2.66618911e-01, 9.79368309e-01, 4.91486078e-02, 1.79660821e-01,
            2.59526006e-01, 1.18172380e-02, 8.56117776e-01, 5.69047533e-01,
            2.26618149e-01, 4.59743669e-01, 2.49983211e-01, 1.98335469e-01,
            5.02006590e-01, 2.32434506e-01, 7.52372617e-01, 9.31146136e-01,
            3.44611996e-01, 3.16541541e-01, 6.27970487e-01, 2.05524782e-01,
            9.09938556e-01, 1.34475621e-01, 9.57497818e-01, 6.08814433e-02,
            8.21203225e-01, 6.77704908e-01, 2.11604285e-01, 7.47502293e-01,
            5.36488317e-01, 6.68349651e-01, 4.15808559e-01, 4.50734776e-01,
            6.15595703e-01, 9.19762327e-01, 3.83088277e-01, 6.39582719e-01,
            4.94882339e-01, 4.09994549e-01, 3.37461967e-01, 7.05128212e-01,
            8.13299193e-01, 2.52582290e-01, 6.34228562e-01, 4.21050816e-01,
            6.01985950e-01, 4.25845504e-01],
           [3.00968372e-03, 2.54182752e-01, 2.67143151e-01, 6.27414651e-01,
            9.63603434e-01, 7.15642862e-01, 2.40487949e-02, 9.44879763e-01,
            3.98285234e-01, 4.67397788e-01, 4.78829296e-01, 4.32287674e-01,
            6.84937482e-01, 2.34680041e-01, 1.48690614e-01, 4.13198413e-01,
            3.43646283e-01, 2.29472247e-01, 8.55011154e-01, 6.18337534e-01,
            6.94396050e-01, 6.35106826e-01, 1.19580524e-01, 7.92073704e-01,
            5.51422947e-02, 4.66614657e-02, 7.26222624e-01, 1.45107096e-01,
            3.18494351e-01, 5.58340712e-02, 2.70614024e-02, 8.63075270e-01,
            4.22202946e-02, 5.28638108e-01, 6.69593813e-01, 8.35029062e-01,
            2.55977306e-01, 4.74062826e-01, 6.41329127e-01, 8.02311107e-01,
            4.90889419e-01, 1.36601777e-01, 9.38832921e-01, 1.32769206e-01,
            6.17844823e-01, 4.31022954e-01, 5.09317906e-01, 2.74034724e-01,
            5.52861026e-01, 1.04424839e-01],
           [5.09130840e-01, 1.11191732e-01, 5.69523846e-01, 2.98006424e-01,
            7.53137318e-01, 7.74417545e-01, 8.08076428e-01, 1.25636263e-01,
            4.17184228e-01, 5.26947314e-01, 2.21750371e-01, 6.92734094e-01,
            8.01542642e-01, 2.54439792e-01, 9.09707435e-01, 1.46527661e-01,
            2.41860567e-01, 5.61130169e-01, 9.89882381e-01, 4.17289507e-01,
            5.43849654e-01, 5.19202113e-01, 2.00671152e-01, 2.31641014e-01,
            9.99252808e-01, 4.90781148e-01, 1.28612614e-01, 3.75219870e-01,
            1.23694998e-01, 4.72090334e-02, 8.35462837e-01, 1.84789770e-01,
            5.56268233e-01, 5.85902535e-01, 2.72172863e-01, 4.80343575e-01,
            7.09086825e-01, 8.44029814e-01, 2.06129591e-01, 8.00175425e-01,
            5.30047394e-01, 3.04658191e-01, 6.34426412e-01, 5.47884901e-01,
            3.16001428e-01, 5.07308928e-01, 4.82682189e-01, 3.33050513e-02,
            8.79484407e-01, 4.93907887e-01],
           [7.22887546e-01, 4.60424458e-02, 5.50645174e-02, 5.72517616e-01,
            6.30597302e-01, 8.45909056e-01, 4.49202020e-01, 2.00334975e-01,
            2.05563953e-01, 7.11116964e-01, 3.89918685e-02, 4.51213943e-01,
            5.93794167e-01, 1.96395169e-01, 7.75917144e-01, 8.53433054e-01,
            6.79200427e-01, 2.49601221e-01, 5.17664097e-01, 1.26392566e-01,
            1.90932852e-01, 8.26646132e-01, 4.16072368e-01, 1.91565390e-01,
            7.77555613e-02, 5.91286329e-02, 5.45122690e-01, 8.31120273e-01,
            9.03241874e-01, 6.28707407e-01, 9.68175002e-01, 3.46013116e-01,
            4.08853292e-01, 7.61106795e-01, 1.38843107e-01, 5.38939794e-01,
            9.22128641e-01, 7.04794363e-01, 7.22885594e-01, 8.19896368e-01,
            8.06882195e-01, 8.88721656e-01, 8.64242045e-01, 8.83438565e-01,
            3.21815489e-01, 6.37086315e-01, 6.80500611e-01, 3.33667398e-02,
            2.51296245e-01, 3.97747281e-02],
           [6.80582558e-02, 3.48867917e-01, 5.12070701e-01, 1.37063577e-01,
            9.03886141e-01, 8.24824456e-02, 7.52606365e-01, 9.14707682e-01,
            1.11413113e-01, 1.85768170e-01, 1.46663118e-01, 9.19234950e-01,
            2.80589136e-02, 1.77516043e-01, 9.11692336e-02, 4.02002145e-01,
            1.59929375e-01, 2.81453083e-01, 6.19749877e-01, 4.79425852e-01,
            8.95315378e-01, 7.92429361e-01, 9.76826424e-01, 9.76427853e-01,
            5.33708622e-01, 8.23082046e-01, 9.02361247e-01, 1.91448089e-01,
            1.20352973e-01, 6.99367457e-01, 8.75767733e-02, 9.31573904e-01,
            6.66273641e-01, 1.54517801e-01, 5.06892853e-01, 5.30220565e-01,
            7.57355432e-01, 1.96201957e-01, 2.07069685e-01, 2.10275671e-01,
            4.46928701e-01, 5.50209110e-01, 3.14404087e-01, 6.87559058e-01,
            6.54896677e-01, 8.50444679e-01, 9.61750063e-01, 6.50639936e-01,
            4.01911200e-01, 3.26131535e-01],
           [6.88868682e-01, 7.35774371e-01, 4.34528404e-01, 8.22063175e-01,
            2.08471452e-01, 1.01168246e-01, 1.46223616e-01, 7.17441057e-01,
            4.64033829e-01, 1.87296476e-01, 9.44689854e-01, 6.40719311e-01,
            1.47128662e-01, 8.61825267e-01, 7.95629160e-01, 7.10037631e-01,
            1.31723561e-01, 3.01955364e-02, 1.18713866e-01, 5.75527383e-02,
            5.75515571e-01, 8.91998517e-01, 5.40889715e-02, 8.56228985e-01,
            6.79087707e-01, 9.57043632e-01, 3.53060098e-01, 3.18918693e-01,
            8.48809784e-02, 2.18214165e-01, 1.98060079e-01, 1.04404818e-01,
            9.24642053e-01, 6.86439816e-01, 7.65896346e-02, 5.74776833e-02,
            3.46117595e-01, 3.52735142e-01, 5.70174849e-01, 8.42072865e-01,
            9.90374619e-01, 2.85972629e-01, 9.82916800e-01, 2.29720739e-01,
            4.96093379e-01, 2.15388413e-01, 2.09083317e-01, 1.48193572e-02,
            7.99991627e-01, 6.18368062e-01],
           [2.14332756e-01, 1.46565592e-01, 6.74120677e-01, 6.75158962e-01,
            3.72233094e-01, 9.70898072e-01, 8.81900805e-01, 2.60248982e-01,
            1.07689016e-02, 4.19663871e-01, 4.15055546e-01, 6.74237442e-01,
            4.26250710e-01, 7.70863234e-01, 1.48543574e-01, 7.97619993e-01,
            5.61641266e-01, 2.25665982e-01, 2.00670835e-02, 5.78143701e-01,
            3.88555609e-01, 2.67805971e-01, 4.64043162e-03, 8.77480891e-01,
            3.85251983e-01, 4.19603975e-01, 5.96399156e-01, 2.03756679e-01,
            8.74071797e-01, 2.44856006e-01, 3.89017651e-01, 6.41016669e-01,
            3.86509118e-01, 3.80314033e-01, 6.50013243e-01, 1.09622345e-01,
            9.68253118e-02, 5.96671841e-01, 8.91136198e-01, 7.19129201e-01,
            9.82442482e-01, 2.68437362e-01, 3.88988530e-01, 3.80945618e-01,
            3.24215853e-02, 7.01045416e-01, 8.07348572e-01, 3.09256355e-01,
            1.10330628e-01, 2.37757246e-02],
           [4.07314995e-01, 6.23814313e-01, 3.10657855e-01, 9.69369228e-01,
            8.54290382e-01, 2.91514683e-01, 3.26246266e-01, 2.66657538e-01,
            4.60254734e-01, 1.66903378e-01, 7.86095320e-02, 9.96580655e-01,
            5.80314630e-01, 8.10559982e-01, 2.09320607e-01, 1.06288219e-01,
            8.04168670e-01, 1.26619728e-01, 6.88982080e-01, 6.03428951e-01,
            3.37469298e-01, 6.83154773e-01, 1.24974366e-03, 1.89481603e-01,
            9.72252074e-01, 1.62812432e-01, 4.60942932e-01, 9.31947987e-01,
            8.59584029e-03, 3.14000566e-01, 3.92436545e-01, 5.78641408e-01,
            8.94039839e-01, 3.22495814e-01, 7.91799336e-01, 7.56238398e-01,
            8.05300495e-01, 7.30900820e-01, 4.37874156e-01, 8.92822439e-01,
            7.23897177e-01, 6.42862955e-01, 1.59368914e-01, 8.53179250e-01,
            5.12702228e-01, 8.65738286e-01, 1.63775543e-01, 4.69664757e-01,
            4.42050686e-01, 2.40456324e-01],
           [1.89232974e-01, 8.61294438e-01, 9.23431526e-01, 3.46840539e-01,
            9.24594506e-01, 7.26856721e-01, 2.39028940e-01, 3.39937237e-01,
            3.58232128e-01, 7.39567065e-01, 3.75268339e-01, 5.65542213e-01,
            5.21416414e-01, 4.01162074e-01, 6.22820659e-01, 3.12993908e-01,
            2.87706538e-01, 4.69333861e-01, 8.98453202e-01, 1.77696589e-01,
            5.71609923e-01, 1.09655719e-01, 9.38571569e-01, 9.65652703e-01,
            8.94713113e-01, 2.01789908e-01, 8.73009889e-01, 7.62406592e-01,
            6.65265894e-01, 3.71892007e-01, 2.28206454e-02, 4.99183914e-01,
            6.08103989e-01, 1.72847387e-01, 2.44742045e-01, 3.94994023e-01,
            1.54988714e-01, 7.00379325e-01, 5.75788236e-01, 9.83863750e-02,
            8.14796283e-01, 5.87117610e-02, 3.44782078e-01, 4.12823262e-01,
            4.33315577e-01, 2.41017871e-01, 9.17566898e-01, 5.31643031e-01,
            8.50996943e-01, 4.43364885e-01],
           [9.90482472e-01, 7.78421569e-01, 9.81286344e-01, 6.53727808e-01,
            3.68864184e-01, 3.13389030e-01, 2.90938577e-01, 6.65767593e-01,
            5.18884145e-01, 4.35739663e-01, 5.44034915e-01, 3.37086711e-02,
            7.92174804e-01, 8.96716175e-01, 7.57851729e-01, 8.52732599e-01,
            8.82971351e-02, 7.55069403e-01, 3.63760028e-01, 8.22339351e-01,
            1.16406692e-02, 8.47633966e-01, 2.64121574e-01, 8.33342220e-01,
            4.60009716e-01, 1.84553485e-01, 8.55387723e-01, 8.17118238e-01,
            4.17935251e-01, 8.10935412e-01, 5.56807746e-01, 2.52673347e-01,
            6.70030474e-01, 5.95122148e-02, 4.73671851e-01, 4.33803282e-01,
            1.41294002e-01, 5.94169965e-02, 7.15228743e-01, 9.53522856e-02,
            8.90669092e-01, 9.12165426e-01, 5.70257204e-01, 2.99217474e-01,
            3.47128470e-01, 9.75168128e-01, 1.34625923e-01, 9.42069519e-01,
            9.76001148e-01, 1.51254238e-01],
           [4.03168054e-01, 3.95690991e-01, 4.90015916e-02, 1.77316880e-01,
            7.39750926e-01, 4.91618402e-01, 7.16056938e-01, 8.48815021e-01,
            5.99150158e-02, 1.52799205e-01, 2.16515417e-01, 6.28681628e-01,
            3.16947148e-01, 9.19609929e-02, 6.82589463e-01, 3.88757498e-01,
            4.23497697e-01, 9.36054219e-01, 2.63130301e-01, 4.61320950e-01,
            8.70614209e-01, 8.15257059e-01, 7.35991228e-01, 7.27697175e-01,
            9.04660373e-01, 9.83861843e-01, 4.99398750e-01, 4.46526811e-01,
            2.30547110e-01, 6.13690243e-01, 7.98055470e-01, 6.51295593e-01,
            2.94839448e-01, 1.09887805e-02, 2.80827236e-01, 4.65303323e-01,
            3.53858425e-01, 8.83293197e-01, 1.13647193e-01, 2.53900762e-01,
            4.06208541e-01, 5.22090706e-01, 3.71352528e-01, 9.78161366e-01,
            5.43947105e-01, 7.41213168e-01, 8.36391655e-02, 8.65328843e-01,
            9.43465721e-01, 1.35656045e-01],
           [7.79004061e-01, 6.97893617e-01, 5.66305257e-01, 6.27112500e-01,
            2.94437708e-01, 9.33351256e-01, 3.83361437e-01, 8.39088308e-01,
            9.05291708e-01, 4.37490654e-01, 7.23489939e-01, 5.04964350e-01,
            8.92014614e-01, 6.56463506e-01, 6.58420491e-02, 7.14017913e-01,
            4.25072788e-01, 1.00761092e-01, 5.47661996e-01, 6.81529843e-01,
            5.44666616e-01, 4.77917478e-01, 1.96379283e-01, 1.36731134e-01,
            2.91615629e-01, 2.70237439e-01, 9.34585008e-01, 4.33885969e-01,
            5.14261936e-01, 7.75462862e-01, 7.08908495e-01, 5.79824247e-01,
            2.30575071e-01, 4.86472608e-01, 4.28397299e-01, 6.55590007e-01,
            4.60494966e-01, 2.22568150e-01, 3.60020503e-01, 9.09277739e-02,
            9.20812106e-01, 9.84244022e-03, 4.46825420e-02, 6.17487261e-01,
            8.09600854e-02, 8.00100980e-01, 2.88511312e-01, 7.41628329e-01,
            4.33634865e-01, 5.35114849e-01],
           [6.26121371e-01, 4.91955020e-01, 5.87384693e-01, 3.56949008e-02,
            4.10578162e-01, 6.68803989e-01, 9.47424878e-01, 3.46864689e-01,
            3.89979756e-01, 5.54772920e-01, 9.14836293e-01, 9.24236026e-03,
            4.51421925e-01, 5.53389329e-01, 8.35928398e-02, 1.87181451e-01,
            9.64022648e-01, 8.12808893e-01, 7.25301861e-01, 6.57469017e-01,
            2.55638990e-01, 5.14077141e-01, 7.57308725e-01, 4.36858356e-01,
            5.75110579e-01, 5.56969553e-01, 2.60245449e-01, 5.37752654e-01,
            9.55558246e-01, 6.19755793e-01, 5.25941873e-01, 5.96109822e-01,
            5.72884008e-01, 6.56443197e-03, 2.14980056e-01, 1.86122619e-01,
            7.53267313e-01, 7.19209356e-01, 9.72341144e-01, 4.48036759e-01,
            5.07992571e-01, 1.66642089e-02, 5.45422435e-01, 7.42916772e-01,
            1.42832266e-01, 5.41663915e-01, 3.70217330e-01, 1.23395860e-01,
            7.14744522e-01, 5.32380707e-01],
           [8.53060845e-01, 8.89868357e-01, 4.92233622e-01, 9.21376624e-02,
            3.32904750e-01, 5.99252495e-01, 8.35415400e-01, 1.01449014e-01,
            8.45199949e-02, 8.28393410e-01, 9.60338124e-01, 8.70991096e-01,
            9.49444725e-01, 2.57270394e-01, 5.96272335e-01, 8.67143703e-01,
            6.93525858e-01, 3.89976918e-01, 1.97752276e-01, 3.51479336e-03,
            1.98850959e-01, 7.15290602e-01, 9.46090263e-01, 6.72586527e-01,
            6.74767016e-01, 9.35034671e-01, 3.81249065e-01, 5.70280153e-01,
            1.10508919e-01, 7.55895046e-01, 5.81063332e-01, 4.87136330e-01,
            7.20992908e-01, 2.97851774e-01, 2.71516470e-01, 9.00410626e-01,
            5.22963654e-01, 1.57930660e-02, 6.26126273e-01, 2.33563869e-01,
            7.99212842e-02, 3.31294846e-02, 6.73446174e-01, 8.56326738e-01,
            7.17222399e-01, 9.82680061e-01, 6.99817880e-01, 5.80988149e-01,
            8.65116696e-01, 6.01000908e-01]])



But, we don't want to do that for all arrays, it would be a hassle. To go back to the default settings:


```python
np.set_printoptions(edgeitems=3, infstr='inf',
linewidth=75, nanstr='nan', precision=8,
suppress=False, threshold=1000, formatter=None)
```


```python
x
```




    array([[0.90949393, 0.02398655, 0.2555127 , ..., 0.24620216, 0.58969889,
            0.0321986 ],
           [0.87734561, 0.65871121, 0.90251625, ..., 0.10830479, 0.17041433,
            0.93544042],
           [0.9007613 , 0.42407472, 0.04579672, ..., 0.53241639, 0.39737813,
            0.15838192],
           ...,
           [0.77900406, 0.69789362, 0.56630526, ..., 0.74162833, 0.43363487,
            0.53511485],
           [0.62612137, 0.49195502, 0.58738469, ..., 0.12339586, 0.71474452,
            0.53238071],
           [0.85306084, 0.88986836, 0.49223362, ..., 0.58098815, 0.8651167 ,
            0.60100091]])



We can also use the following inside a context manager so that it doesn't affect the rest of the code. Notice the difference, "set_printoptions" vs "printoptions":


```python

with np.printoptions(threshold=np.inf):
    print(np.random.random((50,50)))
```

    [[7.84172118e-01 8.41388497e-01 1.41669032e-02 5.75910864e-01
      1.54315468e-01 3.65184693e-01 8.32620100e-01 7.82661429e-02
      2.05213868e-01 8.38190680e-01 9.94762545e-01 1.90082997e-01
      9.67614547e-01 2.86735396e-01 5.78076835e-01 4.92109410e-02
      4.61953287e-01 1.90864068e-02 2.21092681e-02 5.32822024e-01
      4.05850840e-01 6.75050373e-01 5.41246893e-01 1.26259911e-01
      7.96178196e-02 3.02578392e-01 6.95665719e-01 4.34767564e-01
      4.79451200e-01 6.93763424e-01 9.86173786e-01 7.29384490e-01
      4.42474143e-01 5.31298158e-01 3.60836848e-01 8.54271874e-01
      4.66704639e-01 9.82601199e-01 1.10805832e-01 4.32850042e-01
      3.99121697e-01 7.07517465e-01 9.32170671e-01 2.67874588e-01
      1.18401625e-01 5.27529553e-01 6.16991913e-01 1.08190692e-01
      4.38215379e-01 9.95099806e-01]
     [6.31928238e-02 1.92636709e-01 4.99925476e-01 7.98672482e-01
      1.83225094e-01 7.26104207e-01 5.90533301e-01 2.04436972e-01
      7.23525272e-01 2.27152091e-01 4.95306864e-01 1.08570015e-01
      5.71771667e-01 6.99300448e-01 1.07958289e-01 7.16762503e-01
      7.10331926e-01 1.89221492e-01 8.76499501e-01 9.38859256e-01
      3.49628943e-01 8.67435440e-01 3.05025451e-01 6.22081952e-01
      8.34768460e-01 7.88472697e-01 8.15850695e-01 5.94893463e-01
      5.25904453e-01 6.44639833e-01 2.96977486e-02 6.37739164e-01
      7.22177664e-01 8.38424837e-01 3.98413235e-01 6.53879085e-01
      5.47473166e-01 4.12429378e-01 3.24251059e-01 7.80511375e-01
      5.85114574e-01 2.70773599e-01 9.98118766e-01 9.20159997e-01
      2.55225238e-01 1.37047775e-01 3.77365339e-01 7.02601914e-01
      5.15375458e-01 3.64241809e-01]
     [2.55501186e-01 3.60766207e-01 9.81655502e-01 9.94585303e-01
      1.37710450e-01 8.76453991e-01 3.72032086e-01 7.15977130e-01
      6.74674972e-01 9.47478155e-01 2.39660417e-02 6.41118844e-01
      9.96670458e-01 9.28511548e-01 1.78378618e-01 9.71684353e-01
      7.28171619e-01 7.59800201e-01 1.97591678e-01 2.73843431e-01
      8.23972013e-01 3.30996139e-01 2.63387735e-01 2.53146225e-01
      7.22065687e-01 9.50654407e-01 2.90235376e-01 2.79206926e-01
      3.00276151e-02 1.44803210e-01 3.57774555e-01 6.39250480e-01
      2.09937793e-01 1.49337175e-01 2.31692836e-01 9.35079596e-01
      7.21945598e-01 5.42817021e-01 5.29440689e-01 5.20600845e-01
      3.33856127e-01 3.20952731e-01 8.05520204e-01 2.01505322e-01
      7.02625990e-01 9.15365683e-01 9.93631773e-01 9.81689052e-01
      8.02897048e-01 3.63103712e-01]
     [5.43589028e-01 7.86359967e-01 4.66275175e-01 5.51278514e-01
      9.86725522e-01 7.37679319e-01 4.61993096e-01 1.04317591e-02
      7.92307954e-02 2.42179252e-01 8.05764039e-01 6.61150255e-01
      2.45374921e-02 5.24549035e-01 5.46347765e-01 2.40952545e-01
      3.48532767e-01 1.98152746e-01 6.70769072e-01 9.98459040e-01
      3.32122254e-01 1.38765143e-01 5.23116757e-02 1.29000824e-01
      1.10366111e-01 7.52481641e-01 3.93055880e-01 6.46686030e-01
      1.06861653e-01 4.33879473e-01 3.06476933e-01 8.67991273e-01
      6.92619154e-01 3.15021326e-01 1.83267203e-02 4.48105007e-01
      3.83181669e-04 9.78633585e-01 6.75309897e-01 6.82260849e-01
      5.43612558e-01 4.54948109e-01 2.02677094e-01 3.35960120e-02
      2.58358751e-01 5.52406458e-02 2.45593771e-01 7.06410876e-02
      9.28534289e-01 9.04201483e-01]
     [7.29017099e-01 6.62084706e-01 1.70542465e-01 1.42326947e-01
      8.73913104e-02 5.69445211e-01 2.46201390e-01 1.54352650e-01
      5.09970282e-01 4.77978614e-01 7.14726385e-01 7.20983336e-01
      6.33842477e-01 7.84331724e-01 9.28586860e-01 2.13195699e-01
      3.18575387e-01 8.52988965e-01 7.92147002e-01 8.67011001e-01
      7.81668053e-01 2.84791588e-01 1.28434896e-01 9.77152311e-01
      9.44945681e-01 7.95612533e-01 8.75865201e-01 4.15475695e-01
      6.19324178e-01 6.56068800e-01 3.61697816e-01 9.02632890e-03
      4.07318049e-01 8.90548749e-01 3.61867010e-01 5.24882018e-02
      3.92277685e-02 1.15650706e-01 6.58092728e-01 4.68694572e-01
      4.27523421e-01 3.67900580e-01 4.98424584e-01 7.60119257e-01
      5.49291551e-01 2.54885731e-01 3.30742028e-01 7.83830160e-01
      7.30686899e-01 7.23145074e-01]
     [6.23946266e-01 9.14470895e-02 2.52379568e-01 5.69013206e-01
      5.93683442e-02 8.87187416e-01 6.82475105e-01 5.77025792e-01
      4.07528377e-01 8.33624024e-01 2.26177690e-01 6.29730542e-01
      2.56765086e-01 8.01159073e-01 1.93167126e-01 8.89031855e-01
      1.46692844e-01 4.84263324e-01 7.95675542e-01 2.53724598e-01
      2.59188748e-01 9.26491796e-01 7.78074317e-01 1.41432707e-02
      8.48499689e-01 3.21941142e-01 7.61561248e-01 5.41362079e-02
      1.10463141e-01 7.35687618e-01 2.68266369e-01 1.61006993e-01
      4.52306382e-01 3.53340113e-01 8.23465407e-01 3.70964639e-01
      1.60797449e-01 1.18085879e-01 1.39663735e-02 7.96020787e-01
      4.77115196e-01 5.84026985e-01 7.94733314e-02 2.26642262e-01
      7.89995698e-01 9.37370089e-01 6.44774336e-01 1.17529641e-01
      8.33925293e-01 2.52328967e-01]
     [4.90702293e-01 1.29392516e-01 1.30500046e-01 5.69202582e-02
      7.44194721e-01 2.94123638e-01 4.19363197e-01 9.55912381e-01
      6.16053141e-01 9.25483635e-01 7.42216043e-01 1.87912313e-02
      8.29583488e-01 5.10368003e-01 6.32700297e-01 9.52930000e-01
      5.92807762e-01 1.65469948e-01 7.84852106e-01 9.34565171e-01
      2.87572661e-01 8.79089886e-01 3.93175232e-01 4.24697105e-01
      3.99794766e-01 8.93362705e-01 8.12288817e-01 7.43201108e-01
      7.36478143e-01 2.42122336e-01 4.25622640e-01 5.51045961e-01
      1.69726710e-01 3.23719987e-01 9.28483710e-01 6.23365042e-01
      9.24503872e-03 6.45242462e-01 8.19466132e-01 9.53857856e-01
      5.80935024e-01 1.37164549e-02 6.49104365e-01 8.00390355e-01
      3.33060041e-01 7.54023391e-01 6.34330483e-01 6.32628540e-01
      1.55862377e-01 2.22137658e-01]
     [7.37623105e-01 4.81538608e-01 9.10780619e-02 2.63919329e-01
      1.60063254e-01 8.29787279e-01 4.17750868e-01 5.96918922e-01
      4.28856469e-01 4.96985017e-01 3.80042580e-01 9.50373516e-01
      4.08392698e-01 6.44989647e-02 6.10175978e-01 2.78903784e-01
      8.05846977e-01 5.42748940e-01 4.10673369e-01 4.17034166e-02
      4.99102689e-01 3.43212326e-01 1.13012268e-01 1.52314503e-01
      3.50160795e-01 8.91644605e-01 1.45953182e-01 1.11026712e-01
      7.91674921e-01 8.86013601e-01 9.82722171e-01 5.49800223e-01
      9.77698363e-01 4.24400427e-02 6.61989160e-01 4.98940570e-01
      3.22987806e-02 5.94482232e-01 7.41360891e-01 2.88688124e-02
      5.22050176e-01 1.53377984e-01 2.99956779e-01 6.05842154e-01
      7.92586071e-01 5.52579285e-01 6.44096030e-01 1.03095682e-01
      7.67899270e-01 5.68310778e-02]
     [4.19900120e-01 7.38188376e-01 5.57910358e-01 5.31200567e-01
      3.10894546e-01 4.57013915e-01 1.55849265e-01 4.93762601e-01
      4.90404958e-01 5.01709388e-01 3.57808549e-01 9.69133595e-01
      2.72415895e-01 5.04582615e-01 5.00554620e-01 4.67565545e-01
      5.59231810e-01 1.58714830e-01 4.39180266e-01 4.12811683e-01
      7.88765369e-01 1.13995498e-01 6.57987082e-01 3.53587321e-01
      4.71107676e-01 5.41433508e-01 2.64349693e-01 2.74792406e-02
      8.27976295e-01 4.80931696e-01 6.12682531e-01 4.01748534e-01
      9.73785936e-01 9.31508575e-01 2.81857347e-03 1.13846024e-03
      6.73700192e-01 1.92406431e-02 1.19232231e-01 6.21758212e-01
      6.49077872e-01 2.02297966e-01 6.28652094e-01 8.75136931e-01
      9.60545289e-01 5.51326170e-02 8.97856767e-01 7.09248412e-01
      4.49764294e-01 1.21790034e-01]
     [7.49303222e-01 7.42858343e-01 1.99674433e-01 2.55318893e-01
      3.72303336e-01 4.31153767e-02 3.74557429e-01 3.46644338e-01
      5.14343224e-03 1.64240092e-01 5.17509525e-02 7.56741605e-01
      8.17272873e-01 9.41993661e-01 2.49995311e-01 8.40013982e-01
      9.33068002e-01 5.66055953e-01 6.57125455e-01 2.98478787e-02
      2.74839120e-01 5.72678575e-02 3.53227715e-01 5.79770074e-01
      3.68114302e-01 7.38801664e-01 9.71997609e-01 1.53478548e-01
      1.99085894e-01 4.72046248e-01 7.74483372e-01 7.70458060e-01
      2.07268369e-01 9.99063310e-01 9.03125263e-01 4.10688015e-01
      9.60247648e-01 5.94356088e-01 1.90667300e-01 8.67013740e-01
      3.96079677e-01 4.26449197e-01 7.27922716e-01 4.25067269e-01
      4.93385343e-01 4.64465659e-02 8.64706939e-02 8.27634081e-01
      7.94994489e-01 7.07413443e-01]
     [8.28637343e-01 9.10470118e-02 7.35766558e-01 8.48109810e-01
      6.34887640e-01 2.02880411e-01 6.35709490e-01 3.52646118e-01
      4.89765118e-01 9.14570215e-01 2.00174978e-01 9.64010771e-01
      2.76524953e-01 3.10990024e-01 8.20797785e-02 1.93139635e-01
      6.75875787e-01 2.33277071e-02 8.97995149e-01 1.27860337e-01
      9.27547862e-01 1.32632742e-01 2.39662609e-01 3.17167278e-01
      5.24196580e-01 2.03890883e-01 5.47883274e-01 1.12817715e-01
      8.66680417e-01 1.14428548e-01 4.02889601e-01 3.22980380e-01
      4.01934189e-02 9.19547619e-01 7.17232411e-01 9.21629663e-01
      6.02208349e-01 6.57779722e-01 1.59219378e-01 1.57157172e-01
      8.10519973e-01 9.91587947e-01 4.00503215e-01 4.56785353e-01
      1.84365820e-01 5.72172891e-01 1.81194034e-01 9.03873164e-01
      6.51892077e-01 5.78621932e-01]
     [5.40758682e-03 9.27997732e-01 8.63829431e-01 5.99608143e-01
      1.69039405e-01 9.68962941e-01 2.47404234e-01 9.98722700e-01
      3.08141484e-01 8.97663304e-01 7.82530712e-01 2.27267985e-02
      3.49335959e-01 8.16289917e-01 3.36954829e-01 6.47657569e-01
      7.79498259e-01 3.54720295e-02 6.38738204e-01 6.29540742e-01
      8.62480729e-01 1.80761770e-01 4.14096532e-02 2.94411563e-01
      5.57920490e-01 3.64498309e-01 8.61167433e-01 6.57665653e-01
      2.00461743e-01 7.03898654e-01 4.06076522e-02 5.71405040e-01
      5.47706686e-01 6.99526013e-01 8.44165878e-02 2.62831893e-01
      8.30281860e-03 1.64897106e-01 4.94499808e-01 9.15030598e-01
      6.11359643e-01 3.21219834e-01 1.45114780e-01 5.85972325e-01
      3.34437276e-01 9.04963581e-01 3.30253469e-01 6.48130638e-01
      5.36063663e-01 2.20748587e-02]
     [1.53662862e-01 9.76859219e-02 4.99816541e-01 7.16090572e-01
      3.91151474e-01 1.27623508e-01 6.86539333e-01 4.17274985e-01
      1.02102006e-01 1.11418403e-01 4.03808787e-01 2.45694417e-01
      6.27808759e-01 6.81751156e-01 2.79187145e-01 6.35455777e-01
      9.32954121e-02 6.40494206e-01 5.96958279e-02 9.44461827e-01
      7.07523907e-01 8.77768098e-01 3.28273698e-01 5.22011130e-01
      3.48535865e-02 8.02808218e-01 5.76041688e-01 7.16281019e-01
      6.08762625e-01 8.18494648e-01 8.19809904e-01 9.74194061e-01
      7.55146258e-01 5.06182730e-01 6.77197082e-01 4.78952294e-01
      5.24328027e-01 9.78462956e-01 9.38131748e-01 1.35851085e-01
      8.11353752e-01 9.67337908e-01 9.89266079e-01 7.17830107e-01
      4.56623589e-01 2.08673152e-01 1.10244004e-01 5.61259083e-01
      2.92091043e-01 2.92139855e-01]
     [9.20211223e-01 9.69885864e-01 8.98714111e-01 4.94206083e-01
      5.91193425e-01 2.02205724e-01 6.21562656e-01 1.36265545e-01
      2.75232056e-01 8.35467702e-01 4.49915995e-02 5.13102702e-01
      4.95395067e-02 7.10386766e-01 2.69311421e-01 4.36785231e-01
      9.45271271e-01 7.19140442e-01 1.24822515e-01 7.56531965e-01
      1.93913830e-01 1.11507256e-02 3.09688707e-01 2.60567570e-01
      8.80400090e-01 4.25404768e-01 6.86678447e-01 8.28028505e-01
      6.72822825e-01 1.26504068e-02 5.37366221e-01 6.32527881e-01
      2.72215080e-01 3.92149714e-01 1.43655070e-02 8.67272875e-01
      4.69023168e-01 1.68824550e-01 9.97842881e-01 6.97937796e-01
      5.06324840e-01 3.53518803e-01 7.12359581e-02 4.27599365e-01
      9.78435768e-01 3.47979947e-01 8.14087903e-01 6.34215834e-01
      4.18668494e-01 8.07283459e-01]
     [9.20326191e-01 9.19345025e-01 1.12223070e-01 6.86001410e-01
      3.90120260e-01 6.37375052e-01 6.99866218e-01 8.87009366e-01
      7.68269123e-01 1.04875392e-01 7.05100606e-01 6.03587241e-01
      1.92794616e-01 9.94354741e-01 2.39385365e-01 3.43691887e-01
      1.26428465e-01 2.11066095e-01 9.83262781e-01 2.98376070e-01
      6.39917410e-01 8.19637619e-01 3.49145346e-01 3.21876834e-01
      6.77475709e-01 9.92960416e-01 8.73498911e-01 8.20189739e-01
      7.60509312e-01 8.88542838e-03 4.28318769e-01 3.27577011e-01
      1.27865221e-01 7.32104633e-01 1.96334900e-01 4.26632445e-01
      4.19392276e-01 5.68528437e-01 5.56688743e-01 1.42507129e-01
      3.65223358e-01 2.23354171e-01 6.82312340e-01 3.87129604e-01
      3.34848008e-01 8.48511241e-01 8.51804241e-01 7.72713937e-01
      2.43540679e-01 5.25144181e-01]
     [7.45191984e-01 7.60651862e-01 8.40393024e-01 1.62116873e-01
      9.45529940e-01 5.64777841e-01 1.40644864e-01 8.45732214e-01
      9.43461626e-02 3.93460169e-02 6.26449086e-01 1.60795783e-01
      2.45320373e-01 2.32438340e-01 6.83768712e-01 3.10587226e-01
      8.07292933e-01 5.56246019e-01 9.73182356e-01 1.96556584e-01
      7.90767129e-02 6.50101677e-01 6.20359404e-03 2.24172393e-01
      6.38710693e-01 2.84743539e-01 5.07373187e-01 9.45739858e-01
      3.03329517e-01 8.38503820e-01 1.79473309e-01 6.73924967e-01
      8.73362941e-01 4.73392877e-01 2.95855134e-01 4.29406973e-01
      9.77797928e-01 3.89314625e-01 1.43303935e-02 7.07950003e-01
      5.89576434e-01 2.91637849e-01 1.14009955e-01 5.07516174e-01
      8.19107226e-01 2.24051919e-01 7.33070865e-01 5.73604454e-01
      9.59766143e-01 3.73203650e-01]
     [3.37739427e-01 9.98469115e-01 9.35493961e-01 1.63536963e-02
      6.42181026e-01 5.14885707e-01 6.09568251e-01 7.60120171e-01
      6.83156778e-01 7.76429532e-01 4.41565161e-01 8.70632241e-02
      5.74690482e-01 4.05409446e-01 1.31297202e-01 5.30231762e-01
      4.61063002e-01 9.48123405e-02 4.49645717e-01 3.15417190e-01
      4.83800865e-01 3.00124999e-01 2.48732107e-01 8.51172259e-01
      5.31007796e-03 4.61196575e-01 1.11927396e-02 2.96514985e-02
      5.64823754e-01 2.91189348e-01 4.77831827e-01 5.06541256e-01
      5.52973421e-01 6.99937517e-01 2.53577035e-02 4.86315035e-01
      4.33122569e-01 6.92674262e-01 5.44584104e-02 3.72801626e-01
      2.57207497e-01 6.38049834e-02 5.33953682e-01 8.56919020e-01
      8.29397698e-01 4.04905680e-01 5.11159925e-01 8.43942786e-01
      6.21229723e-01 9.75612261e-01]
     [6.40451527e-01 4.93274542e-01 1.68021100e-02 2.21860574e-01
      4.55420323e-01 6.87171840e-01 3.43093105e-01 8.10063997e-01
      4.35058710e-01 2.14798944e-01 5.33853611e-01 2.04941019e-02
      1.40916548e-01 9.14898971e-01 5.85647379e-01 4.06049768e-01
      2.50091346e-01 9.91953170e-01 8.98664687e-01 7.25330192e-01
      8.02564637e-01 2.06959567e-01 5.84353340e-01 5.67989193e-01
      2.62451263e-01 4.39862046e-01 3.74177296e-01 3.90818703e-01
      7.82010287e-01 7.17938559e-01 7.08298584e-01 4.30141022e-01
      8.57756922e-01 8.98369716e-01 3.83266594e-01 6.70061438e-01
      1.78210634e-01 5.05568084e-01 4.95894434e-01 5.86590228e-01
      7.37759225e-02 3.24166460e-01 7.33880206e-01 3.72302028e-01
      3.20103846e-01 5.63675191e-01 5.23647235e-01 2.88082956e-01
      5.09056809e-01 8.81211722e-01]
     [6.60778958e-01 7.25863758e-01 9.04002642e-01 2.16145909e-01
      7.46960030e-01 3.48353558e-01 5.11996615e-01 6.58098457e-01
      9.86260953e-03 8.67833486e-02 2.17400362e-01 1.82548167e-01
      5.65201182e-01 8.44315289e-01 9.47486915e-01 5.39549311e-01
      1.64671634e-01 7.73683185e-01 7.61386975e-01 4.78949798e-01
      2.93708510e-01 3.66007833e-01 9.26723151e-01 2.62535887e-01
      5.12273049e-01 6.44663022e-01 7.12381890e-01 7.42840690e-01
      6.87336085e-01 4.24814565e-01 3.69205807e-01 4.82163254e-01
      1.56053898e-01 9.86816472e-01 9.06381883e-01 3.19523820e-01
      6.39715606e-01 6.70452500e-01 8.09401174e-01 3.28095640e-02
      8.49895651e-01 8.48318549e-01 6.39048374e-03 7.06338350e-02
      9.48142236e-02 6.07503049e-01 4.32044628e-01 2.87028996e-01
      3.78012294e-01 7.69448544e-03]
     [3.05894620e-01 2.70054289e-01 3.47850296e-01 9.11154422e-02
      4.34223890e-01 9.49270520e-01 6.01905027e-01 2.85096042e-01
      8.21262932e-01 5.77500410e-01 2.82026864e-01 2.92815282e-01
      5.77022233e-01 7.17032067e-01 1.03990260e-01 9.41570975e-01
      5.14093679e-01 6.81403275e-01 9.12756885e-01 3.51962424e-01
      9.99200817e-02 4.94066450e-01 4.87020581e-01 9.20864071e-01
      9.42036456e-01 2.35345213e-01 7.04101849e-01 6.39827793e-01
      5.06618067e-01 8.61034541e-01 8.72188234e-01 1.58440652e-01
      4.92040650e-01 4.58913498e-01 3.26247809e-01 3.78914147e-01
      2.91410276e-01 2.06814276e-01 1.52335478e-01 1.29741493e-01
      1.28405283e-01 7.80392081e-01 1.44473640e-01 2.11435085e-01
      7.22722707e-01 4.01356915e-01 1.76605791e-01 1.12922975e-01
      5.90618081e-01 5.87459350e-01]
     [6.96673018e-01 8.47642380e-01 5.84193213e-01 7.91516543e-01
      5.88264370e-01 8.52389618e-01 8.80098511e-01 2.68234967e-02
      5.68076396e-01 1.52915213e-01 7.77528744e-01 6.18861573e-01
      1.57278645e-01 6.77061201e-01 8.96936380e-02 9.58175078e-01
      7.89095931e-01 1.37542933e-02 1.10026876e-01 2.20596658e-01
      2.09559629e-01 8.52761645e-01 5.27058046e-01 2.80907498e-01
      2.67634376e-01 9.01256016e-01 9.78265414e-01 3.26078528e-01
      5.36728058e-01 5.80693574e-03 2.02549288e-01 4.85501102e-01
      3.52815877e-01 8.81807751e-01 1.55372645e-01 5.98645648e-01
      8.55866705e-01 6.53992425e-01 1.32782806e-01 3.34724246e-01
      3.74238197e-01 7.95700440e-01 6.24417409e-01 4.94964826e-01
      9.58278789e-01 6.58984927e-01 9.87966589e-01 9.49989443e-01
      5.95619265e-01 8.08360093e-02]
     [4.76651643e-03 9.31945169e-02 5.81150312e-01 7.00902155e-02
      7.97996711e-02 9.14483555e-01 8.15968655e-01 3.80959032e-01
      7.35599681e-01 9.17919798e-01 5.69963539e-02 1.04497842e-01
      3.10791233e-01 1.61998103e-01 3.12425358e-01 7.29497482e-01
      3.85349782e-01 5.32545963e-01 4.18950891e-01 4.89775890e-01
      6.57925991e-01 7.80979976e-01 6.90973432e-01 3.04932542e-01
      5.48779461e-01 2.64219740e-01 9.21728069e-02 4.30813910e-01
      6.12314419e-01 3.33664308e-01 7.56669130e-01 9.97090345e-01
      6.96409046e-01 1.20790207e-01 1.63236279e-01 9.17118559e-01
      8.78084263e-01 4.80876903e-01 3.20576179e-01 9.84821466e-01
      7.56348109e-01 7.50323199e-01 1.67979191e-01 6.53010776e-01
      6.54456076e-01 7.08370183e-02 1.73492102e-01 1.57642655e-02
      7.31822696e-01 9.01464368e-01]
     [3.54699030e-01 7.29949216e-02 5.72500619e-01 5.25151131e-01
      2.54712861e-02 2.52426788e-01 4.58867183e-01 5.63675413e-01
      9.05702882e-01 6.12485472e-01 9.11319674e-01 1.38713274e-01
      6.84277790e-01 5.51416180e-01 2.30524706e-01 3.95381686e-01
      1.76966560e-01 2.01534288e-01 2.95999642e-01 3.15103171e-01
      8.14265652e-01 1.83180470e-01 4.31124498e-02 5.63535532e-01
      9.00009503e-01 9.22340782e-01 8.05830487e-01 1.34846582e-01
      1.68542926e-01 5.83935679e-01 7.50822839e-01 9.31921308e-02
      4.57171654e-01 2.62243319e-01 1.55632089e-01 7.70095273e-01
      9.00754372e-01 7.35830804e-01 1.97186298e-01 4.20223172e-01
      8.71200153e-01 2.95640959e-01 3.31345490e-01 7.04079293e-02
      5.60051444e-01 7.84384647e-01 8.57780029e-01 5.84097448e-01
      7.18221550e-01 1.63860811e-01]
     [2.82324811e-01 7.59870995e-01 3.70106285e-01 8.58299530e-01
      1.20835680e-01 9.36479253e-01 7.19224223e-01 8.66944181e-01
      5.97482341e-01 3.18164636e-01 2.42541549e-01 8.22442041e-01
      2.33512909e-01 9.15904478e-01 8.54884668e-01 4.64265482e-01
      6.02794182e-01 5.50348994e-01 3.08703489e-01 3.70841289e-01
      8.17004439e-01 8.14672312e-01 6.20690678e-02 6.11108008e-01
      3.09103671e-01 2.10463387e-02 4.24880495e-01 4.06728544e-01
      5.25867843e-01 9.15393190e-01 2.44393591e-01 9.43797161e-02
      6.76017511e-01 6.07698383e-01 7.21156884e-01 3.57978582e-01
      8.99231792e-01 8.70971083e-01 3.37833868e-01 9.91233268e-01
      2.97017810e-03 9.70350975e-01 3.88985934e-01 1.16767373e-01
      8.93957351e-01 5.75928781e-01 4.83861502e-01 8.73406599e-01
      4.62555090e-01 3.11572312e-01]
     [9.50502074e-01 4.85002952e-01 4.19439737e-01 6.15063692e-01
      6.92984072e-01 9.37802584e-01 2.95822127e-01 1.76003021e-01
      3.99199373e-01 5.83497220e-01 8.94722465e-01 4.47221092e-01
      9.68183959e-01 2.66633052e-01 1.38982387e-01 2.47953792e-01
      7.75711764e-01 9.88414656e-01 1.62800865e-01 7.14134711e-01
      2.51395976e-01 4.83020044e-01 4.78080469e-01 2.75284578e-01
      5.92174649e-01 6.07761822e-01 3.61948614e-01 3.62609561e-01
      7.67397964e-01 9.25025289e-01 6.26627423e-01 5.31756847e-01
      1.26509340e-01 3.66356459e-01 1.13667391e-01 2.57627247e-01
      5.07702151e-01 9.81203660e-01 2.88939238e-02 8.65830929e-01
      2.09716857e-01 4.05039675e-01 5.45142798e-01 4.04384822e-01
      9.81952668e-01 2.29718532e-01 6.53011716e-01 6.64438118e-01
      5.39721764e-01 3.89822335e-01]
     [5.71620122e-01 7.50658594e-01 5.46502467e-01 7.35318375e-02
      4.28713204e-02 5.44121219e-01 1.25355958e-01 6.69263405e-01
      2.15964352e-01 7.17366484e-01 7.02271682e-01 9.25498149e-01
      3.28204398e-01 5.07887755e-01 6.16745411e-01 6.27510870e-01
      1.22434547e-01 3.85850010e-01 8.64787174e-01 1.64952820e-01
      5.49711211e-01 3.91459176e-01 4.43964690e-01 1.58988846e-01
      9.65534500e-01 7.83291167e-01 2.25055317e-01 1.64087730e-01
      7.94866382e-01 9.55798640e-01 5.88566831e-01 2.11855174e-02
      2.83945516e-01 2.74574928e-01 3.80073022e-01 2.43535925e-01
      4.89006456e-01 6.50505887e-01 1.14204858e-01 5.32294122e-01
      6.55253661e-01 5.57104438e-01 8.81344355e-01 7.46275561e-01
      8.45784028e-02 5.25719093e-01 6.13782934e-01 7.86962656e-01
      4.46127410e-01 4.84558990e-02]
     [1.73682264e-01 5.90202885e-01 8.55229189e-01 4.72744951e-01
      2.83620047e-01 5.76523320e-01 7.47617221e-01 7.13154236e-01
      7.44799023e-01 1.63747036e-01 9.73509483e-01 4.11021601e-01
      5.39422528e-01 9.24679250e-01 3.69105381e-01 2.66761281e-02
      6.91670536e-02 8.03812193e-01 5.93855483e-01 7.27185820e-01
      8.46218512e-01 1.35073226e-01 8.73912579e-01 5.99399891e-01
      9.72259795e-01 3.50519483e-01 5.78978882e-01 8.42532110e-03
      2.31141122e-01 6.25266014e-01 1.73796337e-01 6.55395460e-01
      3.96902080e-01 3.21790298e-01 1.42642747e-02 4.04307161e-01
      5.45673956e-01 8.87770337e-01 3.75662762e-01 9.80913739e-01
      8.24342566e-01 2.73169543e-01 5.70656591e-01 3.09983590e-01
      4.04086850e-01 4.11152941e-01 2.49979543e-01 4.74099668e-01
      6.98599198e-02 9.95718267e-01]
     [7.81264419e-02 9.36873601e-01 8.82374407e-01 8.98607351e-01
      7.51166341e-01 4.83727104e-01 1.22173579e-02 1.53527791e-01
      5.23485255e-01 1.17266891e-01 4.65555300e-01 6.52254876e-01
      8.56935936e-01 9.96175541e-01 2.13156174e-01 5.74735786e-01
      1.08922209e-01 1.60362021e-01 6.25413649e-01 8.60971047e-01
      4.30120541e-01 7.92567318e-01 9.73352244e-01 4.61851860e-01
      4.63441601e-01 6.20326890e-01 6.84199651e-01 2.77393101e-01
      2.14848336e-01 6.74853492e-01 8.69416519e-02 5.03760582e-01
      2.47552087e-01 4.22021591e-01 6.07115133e-01 6.29189952e-01
      9.75866187e-01 1.39499186e-01 6.65426875e-01 2.98355001e-01
      6.44224573e-01 6.98051347e-01 1.61082182e-01 2.60884281e-01
      9.35503550e-01 4.68931031e-01 6.59739090e-02 8.55988318e-01
      7.09846812e-02 8.17268160e-01]
     [7.86199480e-01 3.30023814e-01 8.53823655e-01 5.27530609e-01
      2.06803910e-01 4.71299603e-01 6.47783748e-01 5.99371274e-01
      5.14665329e-01 3.05114734e-01 7.23644390e-01 8.91177657e-01
      8.79509837e-01 3.37702711e-01 6.89023680e-01 5.14140887e-02
      8.42318717e-02 9.57688882e-01 1.61580697e-01 8.96499978e-01
      1.54687805e-01 1.36238597e-01 7.97878906e-01 5.53482003e-01
      2.49737555e-01 8.09791629e-01 7.50387942e-01 9.17391781e-01
      9.39364336e-01 9.28762840e-01 3.36504138e-01 9.11844378e-01
      2.87731227e-02 9.61202117e-01 8.52593467e-01 3.90065394e-01
      8.54677760e-01 2.88589587e-01 9.18998635e-01 1.86512155e-01
      8.30185484e-02 7.89459972e-01 4.02495591e-01 1.62482690e-01
      9.45591683e-01 9.95502804e-01 1.82799659e-01 8.18627333e-01
      4.82888998e-01 6.97639274e-01]
     [4.43746076e-01 1.47395538e-01 4.85696832e-01 4.76340070e-01
      3.47475047e-01 1.07622502e-01 6.05799377e-01 9.97937991e-01
      5.76186406e-01 1.57279891e-01 2.60791887e-01 2.87898914e-01
      1.36659949e-01 5.35644626e-01 2.61328757e-01 1.57189286e-01
      6.52759844e-01 2.57737941e-01 7.61438398e-01 8.55682681e-03
      7.46414538e-01 8.69334656e-01 1.67921354e-01 6.02738573e-02
      1.14324599e-01 2.34002146e-01 9.24979294e-01 3.85402998e-01
      6.87899514e-01 3.88558983e-01 2.20324126e-01 3.61996496e-01
      7.31275270e-01 6.58211010e-01 3.80073062e-01 8.46776436e-01
      3.16255199e-01 7.78833510e-01 4.20279806e-01 8.65168538e-01
      8.10735357e-01 9.30135797e-01 8.20241593e-01 5.27509207e-01
      3.29529666e-01 7.67454257e-01 3.47014394e-01 9.44920192e-01
      2.84169705e-01 5.47325191e-01]
     [7.17768165e-01 3.14645524e-01 1.70377210e-01 5.60663154e-01
      2.98332292e-01 3.62087626e-01 7.32999791e-01 3.19168894e-01
      7.81246217e-01 8.77323170e-01 7.16310463e-01 3.28061095e-01
      4.78446246e-01 9.53393467e-01 4.72065162e-01 8.61974415e-01
      2.12883949e-01 6.10037341e-01 1.11595408e-01 4.82454222e-01
      8.28041524e-01 5.09775394e-01 3.88881442e-01 9.12865174e-01
      7.86315765e-01 9.13909361e-01 6.24960494e-01 1.94665243e-01
      2.22752693e-01 7.16915459e-01 2.46495704e-01 8.74457755e-01
      1.22274866e-01 8.88419754e-01 8.01235765e-01 5.86897965e-01
      2.20440817e-01 3.57627465e-01 2.52103258e-01 6.30633356e-01
      1.32567148e-01 1.13085278e-01 2.35731526e-01 1.10534043e-01
      5.48568946e-01 3.93565522e-01 1.06675635e-01 7.98565322e-01
      3.85339278e-01 4.69416456e-01]
     [1.77948372e-01 6.57043947e-01 7.35411092e-01 4.26346993e-01
      6.62351410e-01 7.01739304e-01 4.19380804e-01 8.34319949e-01
      6.70176795e-01 4.39461946e-01 5.43094842e-01 4.99872295e-01
      4.85690914e-01 9.27499887e-01 5.35475044e-01 5.06762887e-02
      7.54957483e-01 6.32533011e-01 8.89717409e-01 7.24389252e-01
      8.22342055e-01 1.42181365e-01 7.21986459e-02 3.50946924e-01
      9.10903145e-01 4.40634768e-01 6.00630289e-01 6.68090534e-01
      3.24336625e-02 3.51905727e-01 4.13335013e-01 1.17016891e-01
      3.88901588e-01 7.52064635e-03 7.22107440e-01 1.44633130e-01
      5.95181712e-01 5.01553886e-01 7.13536079e-01 4.06070540e-01
      1.31480368e-01 1.97496842e-01 8.21353554e-01 9.74623267e-01
      7.61449691e-01 4.17312024e-01 5.27063570e-01 4.68783182e-01
      7.72076787e-02 7.56036505e-01]
     [2.96760715e-01 4.69524345e-01 1.44358627e-01 7.78564608e-01
      4.03312764e-01 3.63769903e-01 2.60949771e-01 3.25150153e-02
      9.52316311e-01 7.88544721e-01 2.62313598e-01 5.23040718e-01
      6.68233973e-01 1.03182721e-01 7.39957940e-01 8.84531142e-01
      5.00607575e-02 3.66803395e-01 8.65546201e-01 2.85265791e-01
      9.27099016e-01 3.06414766e-01 5.10532993e-01 4.69749232e-01
      2.76977743e-02 4.98469130e-01 4.21065053e-01 2.40373849e-01
      7.26685003e-01 4.90296780e-01 5.33584872e-01 3.06753754e-02
      4.40041623e-01 4.17210277e-01 3.90662366e-01 2.59043021e-01
      8.47411111e-02 6.38573449e-01 1.87022490e-01 9.65943531e-01
      6.11821184e-01 1.47210897e-01 3.30771385e-01 2.36316997e-01
      5.34068592e-01 5.06479973e-01 7.34782495e-01 3.96884427e-01
      5.32246516e-02 9.75648501e-02]
     [6.63049589e-01 8.65900369e-01 6.68081788e-01 6.56260674e-02
      9.18598656e-01 3.76394165e-01 7.88988450e-01 4.33155830e-01
      7.40407831e-01 5.73991468e-01 6.66908967e-01 8.20446308e-02
      1.13907595e-01 1.06282895e-02 3.58585642e-01 5.28259450e-01
      7.61093131e-01 4.06037193e-01 3.41780986e-01 1.10026980e-01
      4.39916224e-01 6.08640830e-01 4.75476635e-01 6.22658709e-01
      4.36612935e-01 7.83987294e-01 6.43030393e-01 6.35021579e-01
      3.56717452e-01 3.13086261e-01 2.40895919e-01 3.58609213e-01
      4.67813520e-02 8.36177213e-01 1.02306183e-01 8.02523985e-01
      6.96448952e-01 8.15031640e-01 2.46759868e-01 6.80138724e-01
      2.61219912e-01 2.54645954e-01 3.64813167e-01 8.17081965e-02
      5.52366912e-01 4.60306666e-02 7.43777069e-01 8.30759559e-01
      1.73363919e-02 2.72000215e-01]
     [3.16001782e-01 4.10801537e-01 4.28734688e-01 2.44782307e-01
      5.71059525e-01 3.92631440e-01 4.44244478e-01 4.84230915e-01
      1.96522381e-01 9.36195928e-01 8.29726406e-01 8.67015943e-01
      3.34181489e-01 4.28350047e-01 4.54648179e-01 6.61273312e-01
      8.05375640e-01 2.26562076e-01 4.93618668e-01 2.11006596e-01
      5.50194267e-01 6.36230644e-01 1.53401160e-01 5.78856933e-01
      1.26531028e-01 1.38910071e-01 3.98584515e-01 9.43731948e-01
      1.18005167e-01 6.21682858e-02 2.76633995e-01 8.76936172e-01
      8.25363331e-01 2.38757973e-01 6.01521018e-01 3.77545262e-01
      3.85247064e-01 7.17268118e-01 5.64645263e-01 3.49164350e-01
      5.76738903e-01 4.12081011e-01 1.71525121e-01 5.40208105e-01
      2.25081724e-01 7.72543351e-01 1.71949525e-01 3.60816910e-01
      7.87628775e-01 5.42810570e-01]
     [2.75217297e-01 5.78787581e-02 7.34126157e-02 7.94233230e-01
      9.35645684e-04 2.11045739e-01 9.50529702e-02 9.39317623e-01
      1.35486136e-01 7.03135379e-01 4.69721378e-02 2.97824456e-02
      9.20776701e-01 9.02356824e-02 1.67083659e-01 5.76247991e-01
      3.86548888e-01 7.32325648e-01 6.20962121e-01 1.16462676e-01
      7.86762848e-01 8.86795372e-01 3.54656661e-01 7.47085438e-01
      6.97208040e-01 8.50314294e-02 6.58282055e-01 3.20191052e-01
      8.14426755e-02 2.10548960e-01 3.22742921e-01 4.98612548e-01
      6.32514519e-01 2.76127699e-01 9.35685219e-01 4.52471976e-01
      9.50001125e-01 9.26825295e-01 7.26712968e-01 5.35964803e-01
      6.71185047e-01 5.50712219e-01 3.78537322e-01 5.35518114e-01
      2.95713084e-01 4.54655685e-01 2.48516631e-01 4.63927978e-01
      2.07682252e-01 3.03591277e-01]
     [9.24500859e-01 6.32976102e-01 4.86826464e-01 6.63026347e-01
      3.97861528e-01 4.33910760e-01 2.85121513e-01 2.87779307e-01
      9.08681931e-01 1.93725163e-02 6.55457028e-01 7.19498839e-01
      6.30360695e-01 3.53412674e-01 6.03609515e-01 7.05023718e-01
      7.65062230e-01 6.16062634e-02 4.12454980e-01 5.77045092e-01
      1.89360538e-02 8.44315712e-01 1.47971558e-01 8.07734751e-01
      3.62703631e-01 5.95247172e-01 4.61454329e-01 6.10824337e-03
      2.46240064e-01 9.81562427e-01 3.70099153e-01 5.63968674e-01
      6.61715330e-01 1.88127092e-02 4.83442443e-01 7.18554539e-01
      7.66167825e-01 4.07789611e-01 4.12199289e-01 3.37131799e-01
      1.47145851e-01 4.09031822e-01 9.80103410e-02 1.44570814e-01
      7.21087856e-01 5.48403565e-01 2.35076627e-01 7.06272040e-01
      2.64111230e-02 6.31304987e-01]
     [9.03174674e-01 8.60485400e-01 2.24700308e-01 4.74519339e-01
      3.11610831e-01 7.35270734e-01 5.10098381e-01 3.85875346e-01
      8.95790449e-01 9.25628835e-01 7.79802762e-01 7.41174770e-01
      8.64354212e-01 2.91477383e-01 7.55380169e-03 1.82333726e-01
      4.53911405e-01 2.62778898e-01 7.30649331e-01 1.53572349e-01
      6.97804649e-02 5.59970755e-01 3.18173967e-01 9.39034899e-01
      1.31273312e-01 3.20703115e-01 6.15636024e-01 1.34907507e-01
      6.45891147e-01 2.59510713e-01 9.93771804e-01 2.45368714e-01
      6.19902237e-01 4.47567596e-01 2.25876539e-01 3.60533665e-01
      3.48347532e-01 9.23733285e-01 9.31833244e-02 1.65050230e-01
      8.83913689e-01 5.67337812e-01 7.02766587e-01 3.31835449e-01
      5.06474916e-02 6.08290872e-01 7.47325907e-01 6.00458468e-01
      3.65394069e-01 5.14460142e-01]
     [2.24468964e-01 1.63617792e-01 9.19664447e-01 6.08978139e-01
      7.65988099e-01 6.07232164e-01 8.59763539e-02 4.73126484e-01
      7.89812102e-01 7.21852281e-01 2.30136654e-01 8.63554040e-01
      5.85978581e-01 8.54706846e-01 2.11522009e-01 8.00170758e-01
      4.62314743e-01 3.26468488e-01 9.48726598e-01 9.23006807e-01
      4.72239767e-02 6.49606945e-01 9.39730372e-01 2.35204533e-01
      6.69259804e-01 2.88137884e-01 7.04930251e-01 4.75940682e-01
      2.99011697e-01 5.23065400e-01 2.95784387e-01 9.18369151e-01
      7.38328030e-01 9.40148300e-01 3.03245322e-01 8.08998183e-01
      4.93637727e-01 4.62526620e-01 8.16116449e-01 1.47103903e-01
      7.93561431e-01 3.56749090e-01 2.38557902e-01 1.66603649e-01
      1.52308340e-01 3.40963520e-01 8.05087574e-01 7.73682411e-01
      7.35801509e-01 7.79031170e-01]
     [9.48862647e-01 9.33241035e-01 2.47234725e-01 1.83788796e-01
      2.35705163e-01 9.04734931e-01 6.23208320e-01 6.14321808e-01
      8.21526335e-01 3.37988058e-01 9.43275188e-02 4.15049060e-01
      1.96644939e-01 5.66725978e-01 1.94722034e-01 9.98998095e-02
      3.68680552e-01 1.37588739e-01 5.23732596e-01 2.43601278e-01
      7.57875899e-01 2.97171096e-01 3.76736480e-01 9.88426724e-01
      6.39629310e-01 7.13415097e-01 2.99854449e-01 9.52821891e-02
      5.47950128e-01 1.41309368e-01 1.86276970e-01 7.86788644e-01
      4.27591512e-02 2.10778817e-01 4.26672803e-01 1.21013633e-01
      4.27880574e-01 8.97604537e-01 1.46830026e-01 5.38559956e-01
      4.05504389e-01 6.13155736e-01 7.59320166e-01 7.11432915e-01
      6.16180484e-01 8.49645908e-01 8.30343497e-01 3.54314972e-01
      8.07613193e-01 9.45928376e-01]
     [8.34337372e-01 9.16324468e-01 6.04148577e-01 5.03354548e-01
      4.29769142e-02 9.59475195e-01 2.70017750e-01 9.71351928e-01
      2.21053023e-01 2.75800110e-01 2.16046453e-01 8.84391804e-01
      8.39213104e-01 1.65994303e-01 6.18486849e-01 4.57956663e-02
      4.42384095e-01 2.84190681e-01 4.62256396e-01 1.66926408e-02
      5.18233615e-01 5.33573131e-01 2.62379484e-01 4.99482540e-01
      6.35930280e-02 6.46015798e-01 7.35272063e-01 4.51235633e-01
      7.36448634e-01 4.69053538e-01 7.06574696e-01 7.46821219e-01
      9.31064912e-01 6.32125375e-01 4.99796259e-01 9.51481211e-01
      2.59532692e-01 1.52100464e-01 3.72019636e-01 8.46700059e-01
      3.45895656e-01 6.05246728e-01 9.17304585e-01 6.15485599e-01
      5.08326338e-01 6.28970949e-01 2.87726647e-01 5.36031743e-01
      5.73519978e-03 3.73367279e-01]
     [3.85427290e-01 3.34323264e-01 4.70201930e-01 9.52246400e-01
      7.80335954e-01 4.60749895e-01 2.57394479e-01 4.29998435e-01
      1.68554697e-01 7.31135325e-01 9.62360889e-01 9.03356128e-02
      5.21499218e-01 5.71794837e-01 3.12965338e-01 9.24963291e-02
      7.30208970e-01 8.31820887e-01 5.48983717e-01 8.02386322e-01
      1.87621348e-01 4.70222448e-01 7.00766327e-01 1.92877137e-01
      2.77774173e-01 9.81684294e-01 7.42956276e-01 8.43137869e-01
      4.81996980e-01 7.14328525e-01 6.67667474e-01 2.54810438e-01
      3.38189449e-01 2.41797288e-03 4.22306523e-01 9.55209538e-01
      4.10948143e-01 8.31683462e-01 7.83399549e-01 6.52471047e-01
      6.91905840e-01 4.24177492e-02 5.36284827e-01 4.73087386e-01
      4.05891730e-01 6.45210083e-01 2.17533693e-01 5.45537743e-01
      3.03439181e-01 3.21785631e-01]
     [8.32831606e-01 7.78546841e-01 5.20734193e-01 9.44146031e-01
      7.91272555e-01 9.98033629e-01 4.51617941e-01 2.02247590e-01
      8.54794430e-01 4.46302251e-01 6.39530318e-01 1.57952724e-01
      4.37783830e-01 6.51361962e-01 3.46212894e-01 8.85516497e-01
      2.52232202e-01 3.02708148e-01 5.93169905e-01 9.99797240e-01
      6.17709060e-02 1.28982125e-01 1.96356367e-01 9.41347643e-01
      3.29684673e-01 3.37476074e-01 1.90341206e-01 2.89947134e-01
      2.86711553e-01 6.65517466e-01 4.75581057e-02 9.34100890e-01
      5.57760071e-01 2.53357202e-01 2.98219932e-01 2.13655255e-01
      9.50447067e-02 8.18037001e-01 7.73662391e-01 6.03627991e-01
      7.85810391e-01 7.84883919e-03 6.34082784e-01 4.70366833e-01
      7.77293149e-01 8.91209016e-01 6.67404471e-01 9.13850480e-01
      8.95978071e-01 1.74516780e-03]
     [5.36355592e-01 2.30096907e-01 7.67301978e-01 4.85323440e-01
      3.54846140e-01 2.64019304e-01 1.40619792e-01 2.76991050e-01
      1.52026563e-01 3.71429560e-01 9.30043420e-01 5.83083242e-02
      9.68660124e-01 3.64322036e-01 7.36997758e-03 3.90481647e-01
      4.65991193e-01 7.97600253e-01 2.36862372e-02 7.67113682e-01
      9.87533784e-01 4.26186594e-01 6.59396321e-01 8.51782336e-01
      3.18400790e-01 6.97144912e-01 8.28896108e-01 7.08410218e-01
      7.59840529e-01 2.81710194e-01 5.46464787e-01 3.35702162e-01
      9.71023987e-01 2.33969724e-01 8.79411453e-01 8.60027816e-01
      1.44776202e-01 3.20754513e-01 7.02061535e-01 6.78792419e-01
      9.03459331e-01 2.39245906e-01 8.52161412e-01 5.49052116e-01
      5.12220598e-02 8.20200384e-01 7.24014238e-01 9.73721332e-01
      3.71306313e-01 1.87034125e-01]
     [7.46966642e-01 9.30393628e-01 8.05977041e-01 2.27349768e-01
      2.36451522e-01 6.65286415e-01 8.34000153e-01 8.69128697e-01
      1.83360975e-01 3.22783526e-01 4.54894528e-01 4.94365510e-01
      2.08427197e-01 1.99426730e-01 1.80130774e-02 5.50465409e-01
      4.68972108e-01 1.18643141e-01 3.64468947e-01 1.11237408e-01
      7.70797501e-01 8.53433968e-01 7.03469254e-01 5.53414663e-01
      2.76623582e-01 4.83130790e-01 5.10968813e-01 2.84060863e-01
      8.20940060e-01 7.77604402e-01 7.96999305e-01 1.51109179e-01
      5.32089117e-01 2.96616497e-01 9.35622861e-01 5.01248515e-01
      5.31397802e-01 4.57899692e-01 3.24006960e-01 5.59117793e-01
      5.08878180e-01 6.99239961e-01 2.32807972e-01 8.05388921e-01
      6.43452752e-01 2.12809596e-01 6.97697984e-01 3.87406940e-01
      5.71637770e-01 8.23135560e-01]
     [2.88140468e-01 8.66273372e-01 5.71064307e-01 5.18575029e-01
      7.51042201e-01 9.21720325e-01 2.50384459e-01 8.28970713e-02
      7.06806253e-01 2.44901362e-01 8.20325489e-01 8.68990527e-02
      6.91574618e-01 2.30540401e-01 6.95024183e-01 7.60884599e-01
      9.00007230e-01 4.91385105e-01 3.86458391e-01 6.86150466e-01
      1.08505677e-01 6.83416483e-01 5.58944366e-01 1.07221182e-01
      5.52178045e-01 4.00552260e-01 2.54135300e-01 4.14628015e-01
      9.70571047e-01 5.91840253e-01 4.94313342e-01 7.88217358e-01
      3.29931570e-01 1.35835641e-02 8.35657255e-01 9.01157519e-01
      4.81597049e-01 8.85571343e-01 1.08728513e-01 8.06400530e-01
      3.09278226e-01 9.25845935e-01 4.16327631e-01 4.48872484e-01
      2.18008327e-02 3.14971408e-01 2.01557086e-01 1.48910812e-01
      3.06965961e-02 9.42193754e-01]
     [4.24028390e-01 6.31659602e-01 3.23528662e-01 3.49755065e-01
      3.98397910e-01 2.85225927e-01 7.80975643e-02 5.05828110e-01
      5.24792946e-02 3.64457674e-03 2.85358972e-01 5.50965855e-01
      2.09802290e-01 4.69150100e-01 4.75564161e-01 7.82445231e-01
      4.98921346e-01 2.00816001e-01 7.81997264e-01 8.59338773e-01
      9.34551966e-01 4.04903464e-01 2.47524009e-01 8.93862255e-01
      9.16448555e-01 8.01138744e-02 3.03932928e-01 4.72776421e-01
      3.76222914e-01 2.12333662e-01 7.27969018e-01 9.05129403e-01
      4.62682934e-01 3.95458296e-01 2.77378429e-01 8.40920909e-02
      2.64516168e-01 1.29919034e-01 9.81845749e-01 2.90146298e-01
      1.86838892e-01 1.87652923e-01 7.88331119e-01 1.14787490e-01
      8.88076047e-01 6.80354100e-02 3.98963718e-01 5.85797028e-01
      8.92466927e-01 9.96299659e-02]
     [2.87065379e-01 3.51059442e-01 5.30319060e-01 1.38518689e-01
      3.35419388e-01 9.56986709e-01 9.07806940e-01 6.92612795e-02
      9.29559945e-02 6.58537457e-01 2.84654749e-01 1.84373837e-01
      2.71292361e-01 1.79979082e-01 2.43749125e-03 5.98860729e-01
      1.14767450e-02 9.52228301e-01 4.09125905e-01 3.27650817e-01
      4.81986729e-01 6.04673533e-01 9.81909709e-01 7.41961316e-01
      5.53066796e-01 8.74591871e-01 9.82998653e-01 4.27329617e-01
      9.09515679e-01 2.83985397e-01 1.54112672e-01 5.67853625e-01
      7.90483388e-01 9.22239804e-02 9.42979948e-01 6.05992614e-01
      7.85561804e-01 4.62696373e-01 8.00930158e-01 2.98600717e-01
      3.84854021e-01 4.01603042e-01 6.21318389e-01 9.61914253e-01
      3.53305341e-01 2.53248741e-01 5.38726594e-01 6.91533691e-01
      6.37276308e-01 2.12949070e-01]
     [3.15411336e-01 5.42344693e-01 2.87454384e-01 9.85910134e-01
      5.96078553e-01 1.44455989e-01 2.15002177e-01 6.63999373e-01
      7.57353419e-01 5.45133566e-01 8.59819268e-01 7.01029493e-01
      1.55179021e-01 2.42967883e-01 4.89284436e-01 5.12255770e-01
      7.59440310e-02 2.33310730e-01 3.11107239e-01 9.77046642e-01
      2.20464867e-01 6.50301804e-01 6.25869070e-01 4.59439298e-01
      6.44285538e-01 4.74564652e-01 5.89455542e-01 2.32590515e-01
      8.24245453e-01 1.35426644e-01 1.93395889e-01 5.16348814e-01
      3.64535794e-01 6.52123487e-01 6.13990669e-01 5.81422314e-01
      9.32018955e-01 7.29038497e-01 3.88812980e-01 5.31854053e-01
      4.88117804e-02 8.50496005e-01 4.09320783e-01 4.70363504e-01
      1.55455941e-01 7.07254903e-01 8.76691729e-01 4.10768127e-01
      6.89459470e-01 3.22233833e-02]
     [9.42815125e-01 1.61490441e-01 1.52390621e-01 6.08340083e-01
      8.52476042e-01 6.31164475e-01 1.65746304e-01 1.72706081e-01
      2.28219551e-01 8.22922632e-01 8.26253627e-01 4.86726979e-01
      8.38315039e-02 8.62711762e-01 1.56645999e-01 1.17890940e-01
      9.92522584e-01 9.22037055e-01 8.88267271e-01 7.30676538e-01
      6.16107147e-01 3.69790672e-01 4.25767903e-01 6.52924985e-01
      3.24339318e-01 2.43970629e-01 4.62470238e-01 3.14238234e-01
      9.48267928e-01 7.89653649e-01 2.21745144e-01 4.59907758e-01
      7.81384872e-01 5.48467543e-01 7.78414168e-01 5.42218022e-01
      2.78398471e-02 8.03673480e-01 4.44378145e-01 5.88655786e-01
      1.92256884e-01 7.71941524e-01 1.99025945e-01 4.94543412e-01
      6.76887671e-01 5.98585205e-01 4.82785305e-01 4.23154112e-01
      9.41521563e-01 1.16658508e-01]]
    

#### 50. How to find the closest value (to a given scalar) in a vector? (★★☆)

I didn't really understand the question to be honest, but once I looked at Rougier's solution, everything made sense.


```python
Z = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(Z-v)).argmin()
print(Z[index])
```

    94
    


```python

```
