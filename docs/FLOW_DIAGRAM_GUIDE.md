# Hand-drawn flow diagram - what to draw on paper

Draw this by hand (pen on paper), photograph it clearly, save it as
`docs/flow_diagram.jpg`, and it will show up in the README automatically.

```
 [START]
    |
 [Set seed = roll number]
    |
 [Generate 20x20 grid: random obstacles, random start & goal on FREE cells]
    |
 [BFS check: is goal reachable?] --no--> (regenerate, same RNG stream)
    | yes
 [Initialise swarm: 120 particles, each = 8 waypoints (x,y);
  half near the straight line, half random; random velocities]
    |
 [Evaluate each particle: path = Start + waypoints + Goal
  cost = length + 50 * (# sampled points inside obstacles)]   <-- collision check
    |
 [Set pbest = own position, gbest = best of swarm]
    |
 +-->[Update inertia w (0.9 -> 0.4)]
 |   [v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x); clip v]
 |   [x = x + v; clip to grid]
 |   [Evaluate new cost (length + collision penalty)]
 |   [Update pbest / gbest if better]
 |       |
 |   <iteration < 300 ?> --yes--+
 |       | no
 |   <feasible path found (0 collisions)?> --no--> [restart PSO, new swarm seed]
 |       | yes
 [Output best path, length, collisions]
    |
 [Plot grid + obstacles + start + goal + path (matplotlib)]
    |
 [END]
```
Use boxes for processes, diamonds for decisions (iteration check,
feasibility check) and arrows for flow. Label the loop clearly.
