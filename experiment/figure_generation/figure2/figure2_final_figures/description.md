#Final Results for Figure 2 

### How we plot the graph
1. We use BFS to gather ~100k circuits for each gate count in 
   [58, 56, 54, 53, 52, 51, 50, 48] and around 4.6k circuits
   for gate count [46]. Dataset for gate count 46 has fewer
   circuits because BFS can only find 4.6k circuits with 46
   gates if we do not allow xfers that increase gate count.
   The dataset we collected can be found in 
   `../categorical_generator/final_dataset`.
   
2. Then we run BFS on each dataset to see the minimal number
   of xfers each circuits needs in order to reduce its gate
   count. When running BFS, we still do not allow xfers that
   will increase the gate count. (o.w. the search space will
   be too large)
   
3. Finally, we gather the data and plot distribution of 
   `min #xfer` for each gate count. This shows how the
   difficulty changes along the optimization path.