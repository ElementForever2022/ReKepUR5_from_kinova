# Development Log within PAL Lab
Tony Wang @ PAL Lab


## Experiment
> work as a baseline for Zeromimic 
- [ ] Pouring tea
- [x] Put the green package in the drawer
- [x] Close the drawer
- [x] Open the drawer
- [x] Put down the green package
- [ ] Brew a cup of espresso

## TODO

### 2024-12-17
- [ ] What is different from ReKeP’s own codebase
    - [ ] Plus
        - [ ] better clustering for keypoint extraction
    - [ ] Minus
        - [x] multiple stage: bug
        - [ ] ik cost - doensn’t matter
        - [ ] recovery
- [ ] stage decomposition  
    - store: in robot_state.json
    - load: solely in r2d2_rekep.py
    - question: keypoint following?
- [x] Code-as-Monitor, another work with ReKeP as baseline in Omnigibson, pour tea result is 20, wherweas ReKeP official result is ~70%
- [x] path solver, cannnot generate action sequence?
    - [x] pdb trace to see optimization
    - [x] Reason: the given subgoal constraints are unvalid
    - [x] Solution: 
        - [x] leave it as default error
        - [x] modify the prompt
        - [x] improve environment.py
        - [x] test the subgoal constraints? (add a transform to compute from end effector)

### 2024-11-23
- [ ] add DINO-X into system
    - [ ] prompt free detection
    - [ ] prompt guided detection
- [ ] add more visualization method
- [x] test DepthPro 
    - [x] least square method
    - [x] gaussian process
    - [x] ask GPT for help
    - [x] ask SYZ for help
- [x] Kinemtic
    - [x] MEAM5200 note
    - [x] ECE470 note
- 
```bash
configs/ # rekep only
data/    # temp data
main.py  # original rekep for omnigibson
main_vision.py  # rekep with vision
point-draw/ # hand drawing tool
r2d2.py  # modify for r2d2
r2d2_rekep.py # deploy rekep on r2d2
r2d2.sh   # run r2d2
r2d2_vision.py # deploy rekep with vision on r2d2
vision.sh # run r2d2_vision.py
vlm_query/ # folder store vlm query
```
- [x] Test the new tasks
- [x] Add the new scenes
- [x] Add the new objects
- [ ] Add new camera views
- [ ] 3D visualization of RGBD
- [x] extristic and intrinsic camera calibration
- [ ] 3D bounding box visualization? 
- [x] add grounding dino into system
