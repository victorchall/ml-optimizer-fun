A simple single-parameter simulation of the Adam optimizer.  The purpose is to help an individual build a bit of intuition of how Adam works over multiple training steps, with views for the intermediate values (m, v) and actual learning step applied given a seqeunce of grads as input. 

Try adjusting beta1, beta2 and different sequences of grads, and how that affects the 1st and 2nd moment, and the actual learning step that would be applied to the weight.

[Adam Simulator](https://colab.research.google.com/github/victorchall/ml-optimizer-fun/blob/main/adam_simulator.ipynb)

A quick modeling of what Huber and MSE loss might look like combined by a linear interpolation applied against an independent variable (here, labeled timesteps from 1000 to 0, as it might be used for diffusion models).

[Timestep graph, and Huber and MSE loss interpolation](https://colab.research.google.com/github/victorchall/ml-optimizer-fun/blob/main/huber_mse_interpolated.ipynb)
