# What I want to do?

multiscale prediction in space and time. 
  - Need to break space to low dim, run though time, refine, 


## averageing space
  - start 2x2, with average as the value. 
  - grow 2x2 to 4x4, find error between grow(2x2) and average(4x4)
  - shrink back to 2x2 with average error, find which blocks have an error > theshold
  - refine. blocks that are unresolved will get split in 2 again. Find error and find blocks > threshold. 
  - new predicted block is 4x4, but blocks refined at first block are all same. 
