Implementation of the SIMBICON [paper](http://www.cs.ubc.ca/~van/papers/2007-siggraph-simbicon.pdf). 

<img width="1088" alt="image" src="https://user-images.githubusercontent.com/13502307/202119195-cf9af1f4-2809-49bd-b528-e301e9a0eca5.png">

Here's a [video](https://www.youtube.com/watch?v=ytE2kxJCwqo) of a running biped.


### Simbicon Controller
In order to visualize a motion, specify the target angles and other parameters in a .yml file. (See `settings/config.yml` for an example of how to specify this file.)

Then run ``` python simbicon.py``` with relevant parameters as specified in the `simbicon.py` file. 


Example:
``` python simbicon.py -m jog -p settings/config.yml```

### CMA Optimization 

CMA is used to optimize simbicon parameters (target angles, torso_kp, torso_kd, FSM time interval) for a given target velocity and style.
Look at `cma.py` for details on what parameters to specify before optimization. Running this file will yield a yml file with saved parameters.

To visulize the optimized parameters, you can run the ```simbicon.py``` file with the path to the optimized parameters. 

Example: Here we use the initial parameters for jogging (which is at 1.8m/s) to optimize for faster running.
```python cma.py -lm jog -sm running -sp settings/cma_config.yml -tv 3.5 ```
We can then visualize the result:
```python simbicon.py -m running -f cma_config.yml```


### Acknowledgements
Thanks to [Ben Ling](https://www.cs.ubc.ca/~hyuling/) for the CMA optimization code and [Michiel Van de Panne](https://www.cs.ubc.ca/~van/) for helpful discussions. 
