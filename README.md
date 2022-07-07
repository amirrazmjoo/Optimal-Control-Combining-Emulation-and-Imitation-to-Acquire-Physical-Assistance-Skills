# Optimal-Control-Combining-Emulation-and-Imitation-to-Acquire-Physical-Assistance-Skills
This paper studies exploiting action-level learning (imitation) in the optimal control problem context. Cost functions defined by the optimal control methods are similar to the goal-level learning (emulation) in animals. However, imitating the robot's or others' (e.g. human's) previous experiences (demonstrations) could help the system to improve its performances. We propose to use demonstrations more efficiently by predicting an initialization for the optimal control problems (OCPs) and adding an imitation term to the cost functions. While the predicted initial guess initializes the OCPs close to their local optima, the imitation term guides the optimization, resulting in a faster convergence rate. We test our algorithm in a physical assistive task where a robot should help a human perform a sit-to-stand (STS) task. We define this task as two optimal control problems. The first OCP predicts the human's desired assistance and the other one controls the robot. We have tested our method on different experiments with different conditions for the human in which the robot should quickly solve the two optimization problems exploiting some demonstrations of how it can perform the task. Our proposed method reduced the number of iterations by more than 90% and 70% for the human assistance prediction and the robot controller, respectively, compared to the standard problem which does not take the demonstrations into account.

Links to the paper: [preprint](https://publications.idiap.ch/attachments/papers/2021/Razmjoo_ICAR_2021.pdf), [video](https://www.youtube.com/watch?v=K8fQ43UN92E)