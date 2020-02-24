import numpy as np
from nndl.layers import *
import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #
  npad = ((0,0),(0,0),(pad,pad),(pad,pad))
  xpad = np.pad(x, pad_width = npad, mode='constant', constant_values=0)
  N,C,H,W = xpad.shape
  F,C,HH,WW = w.shape
  H_pad = int(1 + (H + 0 * pad - HH) / stride)
  W_pad = int(1 + (W + 0 * pad - WW) / stride)
  out = np.zeros((N,F,H_pad, W_pad))

  for n in range(N):
    for f in range(F):
      for h in range(H_pad):
        hs = h * stride
        for ii in range(W_pad):
          ws = ii * stride
          out[n,f,h,ii] = np.sum(w[f]*xpad[n,:,hs:hs+HH,ws:ws+WW]) + b[f]

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
  N,C,H,W = x.shape
  H_pad = int(1 + (H + 2*pad - f_height) / stride)
  W_pad = int(1 + (W + 2 * pad - f_width) / stride)

  dxpad = np.zeros_like(xpad)
  dx = np.zeros_like(x)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)

  for n in range(N):
    for f in range(num_filts):
      db[f] += np.sum(dout[n, f])
      for jj in range(H_pad):
        hs = jj*stride
        for ii in range(W_pad):
          ws = ii * stride
          dw[f] += xpad[n, :, hs:hs + f_height, ws:ws + f_width] * dout[n,f,jj,ii]
          dxpad[n, :, hs:hs + f_height, ws:ws + f_width] += w[f] * dout[n,f,jj,ii]

  dx = dxpad[:,:,pad:pad+H,pad:pad+W]
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  HH = pool_param['pool_height']
  WW = pool_param['pool_width']
  stride = pool_param['stride']
  N, C, H, W = x.shape
  Hp = int(1 + (H - HH) / stride)
  Wp = int(1 + (W - WW) / stride)

  out = np.zeros((N, C, Hp, Wp))

  for i in range(N):
    for j in range(C):
      for k in range(Hp):
        hs = k * stride
        for l in range(Wp):
          ws = l * stride

          window = x[i, j, hs:hs+HH, ws:ws+WW]
          out[i, j, k, l] = np.max(window)

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  N,C,H,W = x.shape
  H_pad = int(1+(H - pool_height) / stride)
  W_pad = int(1 + (W - pool_width) / stride)

  dx = np.zeros_like(x)

  for n in range(N):
    for c in range(C):
      for jj in range(H_pad):
        hs = jj * stride
        for ii in range(W_pad):
          ws = ii*stride
          window = x[n, c, hs:hs+pool_height, ws:ws+pool_width]
          m = np.max(window)
          dx[n, c, hs:hs+pool_height, ws:ws+pool_width] += (window == m) * dout[n, c, jj, ii]

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)
  N,C,H,W = x.shape
  running_mean = bn_param.get('running_mean', np.zeros((1, C, 1, 1), dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros((1,C, 1, 1), dtype=x.dtype))
  out,cache = None, None
  if mode == 'train':
    sample_mean = np.mean(x, axis = (0,2,3)).reshape(1,C,1,1)
    sample_var = 1/float(N*H*W) * np.sum((x-sample_mean)**2, axis=(0,2,3)).reshape(1,C,1,1)
    #1
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    xhat = (x - sample_mean) / np.sqrt(sample_var + eps)
    out = gamma.reshape(1,C,1,1) * xhat + beta.reshape(1,C,1,1)
    cache = (x,xhat,sample_mean,sample_var,gamma,beta,eps)

  elif mode == 'test':
    outhat = (x - running_mean)/np.sqrt(running_var+eps)

  else:
    raise ValueError('Invalid ForwardBatchNorm Mode %s' % mode)

  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  x,xhat,sample_mean,sample_var,gamma,beta,eps = cache
  N,C,H,W = dout.shape
  dbeta = np.sum(dout,axis=(0,2,3))
  dgamma=np.sum(dout*xhat, axis=(0,2,3))
  gamma_reshape = gamma.reshape(1,C,1,1)
  beta_reshape = beta.reshape(1,C,1,1)
  V = N * H * W
  dxhat = dout * gamma_reshape

  mdvar = np.sum(dxhat * (x - sample_mean), axis = (0,2,3)).reshape(1,C,1,1)
  dvar = mdvar * -1/2 * (sample_var + eps) ** (-3/2)
  ds = 1/V * np.broadcast_to(np.broadcast_to(np.squeeze(dvar), (W,H,C)).transpose(2,1,0), (N,C,H,W))
  dx1 = dxhat / np.sqrt(sample_var + eps) + 2 * (x-sample_mean) * ds
  dmu = -1 * np.sum(dx1, axis = (0,2,3))
  dx2 = 1/V * np.broadcast_to(np.broadcast_to(np.squeeze(dmu), (W,H,C)).transpose(2,1,0), (N,C,H,W))
  dx = dx1+dx2
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta