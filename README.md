# LfD-Perturbations
 Implementation of Perturbation Analysis for Learning from Demonstration (LfD).

Corresponding paper can be found for free [here](https://arxiv.org/abs/2208.02207), please read for method details. Accompanying video available [here](https://youtu.be/IQDxbhEiNbk).

Several methods exist for teaching robots, with one of the most prominent being Learning from Demonstration (LfD). Many LfD representations can be formulated as constrained optimization problems. We propose a novel convex formulation of the LfD problem represented as elastic maps, which models reproductions as a series of connected springs. Relying on the properties of strong duality and perturbation analysis of the constrained optimization problem, we create a confidence metric. Our method allows the demonstrated skill to be reproduced with varying confidence level yielding different levels of smoothness and flexibility. Our confidence-based method provides reproductions of the skill that perform better for a given set of constraints. By analyzing the constraints, our method can also remove unnecessary constraints. We validate our approach using several simulated and real-world experiments using a Jaco2 7DOF manipulator arm.

<img src="https://github.com/brenhertel/LfD-Perturbations/blob/main/pictures/real_world_experiment/confidence_figure.png" alt="" width="800"/>

This repository implements the method described in the paper above using Python. Scripts which perform individual experiments are included, as well as other necessary utilities. If you have any questions, please contact Brendan Hertel (brendan_hertel@student.uml.edu).

If you use the code present in this repository, please cite the following paper:
```
@inproceedings{hertel2023confidence,
  title={Confidence-Based Skill Reproduction Through Perturbation Analysis},
  author={Hertel, Brendan and S. Reza Ahmadzadeh},
  booktitle={20th International Conference on Ubiquitous Robots (UR)},
  year={2023},
  organization={IEEE}
}
```
