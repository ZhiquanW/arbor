# roadmap
## todo
1. Jan 23, 2023
    - [ ] ~~util method: change spherical coordinate to direction based on the reference axis~~
2. Jan 24, 2023 - 22:39
    - [x] compute rotation matrix from delta rotation $M_d$
    - [x] compute rotation matrix from previous rotation $M_p$
    - [x] get new rotation matrix $M_p M_d$
    - [ ] ~~if viz, then compute axes from $M_d M_p$~~
    - [x] compute new rotation from rotation matrix 
3. Jan 25, 2023 - 12:39
    - [x] compute moving direction (up) 
    - [x] compute positions
    - [x] draw positions
4. Jan 25, 2023 - 22:11
    - [x] sleep nodes 
    - [x] perform indices from indices experiments b = a[10:20] c = b[5:10] ,would change b affect a?
    - [x] grow nodes, sample from active nodes and perform corresponding rotation, 
5. Han 26, 2023 - 22:11
    - [ ] (Optional) move sleep to the end of the step
    - [ ] if no awake vertices, then return
    - [ ] fix "AssertionError: rot_axes.shape=(2, 3) does not match (1, 3)"