# This is the log of developing problems of this repo

## Oct 4th:
### Todo:
1. Figure out the performance gap, Current situation is: Performance drop after first batch. Potential reasons are as following:
   1. Maybe only the first batch is running on GPU.(Witnessing no GPU usage rise after first batch; the performance gap happens to fit the batchsize)
   2. Memory limit?(Personally I don't think that's the case)
   3. Torch inherent property? If that's the case then maybe just live with it.
2. Try the transformer-based method that fuses the temporal message.
3. The "max-episode-len" argument is something that we can change for different scenarios.
4. The "select_action" please make it better and easier to add new algorithms.