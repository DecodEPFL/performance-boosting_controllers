# Performance-boosting Controllers

PyTorch implementation of control policies as described in 
"Learning to boost the performance of stable nonlinear closed-loop systems."


## Installation

```bash
git clone https://github.com/DecodEPFL/performance-boosting_controllers.git

cd performance-boosting_controllers

python setup.py install
```

## Examples: 

The environment consists of two robots that need to achieve some predefined task while avoiding collisions 
between them and the obstacles.
In the first case, the robots need to pass through a narrow corridor, while in the second
case they have to visit different goals in a predefined order.

### Corridor
The following gifs show the trajectories of the 2 robots before training (left), and after the training 
of the performance-boosting controller (right). 
The agents need to coordinate in order to pass through a narrow passage while avoiding collisions between them and 
the obsacles. 
The initial conditions used as training data are marked with &#9675;, consisting of s = 100 samples.
The shown trajectories start from a random initial position sampled from the test data and marked with &#9675;.

<p align="center">
<img src="./gif/olNominal.gif" alt="robot_trajectories_before_training" width="400"/>
<img src="./gif/clNominal.gif" alt="robot_trajectories_after_training" width="400"/>
</p> 

#### Robustness verification
We verify our robustness results regarding model uncertainty. We test the above trained controller on systems with
&pm;10% variance on the mass robots.
Each of the following gifs shows three closed-loop trajectories when considering 
lighter (left) and heavier (right) robots. 

<p align="center">
<img src="./gif/cl10lighter.gif" alt="robot_trajectories_with_lighter_mass" width="400"/>
<img src="./gif/cl10heavier.gif" alt="robot_trajectories_with_heavier_mass" width="400"/>
</p> 

#### Safety and invariance
Using soft safety specifications in the form of control-barrier like losses, we perform a new training of the 
controller in order to promote robots not visiting the subspace above the target. 
We refer to our paper

<p align="center">
<img src="./gif/clBarrier.gif" alt="robot_trajectories_barrier" width="400"/>
</p> 

### Waypoint-tracking
The following gifs show the trajectories of the two robots before training (left), and after the training 
of the performance-boosting controller (right). 
The agents need to visit different waypoints in a specific order while avoiding collisions between them and 
the obstacles.
* Robot Blue: _g<sub>b</sub>_ &rarr; _g<sub>a</sub>_ &rarr; _g<sub>c</sub>_
* Robot Orange: _g<sub>c</sub>_ &rarr; _g<sub>b</sub>_ &rarr; _g<sub>a</sub>_

<p align="center">
<img src="./gif/wp_ol.gif" alt="robot_trajectories_before_training" width="400"/>
<img src="./gif/wp_cl.gif" alt="robot_trajectories_after_training" width="400"/>
</p>



## License
This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by] 

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg


## References
[[1]](...) Luca Furieri, Clara Galimberti, Giancarlo Ferrari-Trecate.
"Learning to boost the performance of stable nonlinear closed-loop systems," 2024.
