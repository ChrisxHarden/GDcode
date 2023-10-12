# This is the log of developing problems of this repo

## Revision:


### Extra Experiments
1. The research presented in [1] explores a topic akin to that of this paper. However, the differences between the two have not been explicitly delineated, and no experimental analysis has been undertaken to compare the efficiency of the methods proposed in this paper with those in [1].
2. The LSTM employed in this paper appears to be outdated. It would be advisable to consider more advanced time series frameworks, such as Transformer-based [2] or Graph-based approaches. These cutting-edge frameworks have demonstrated superior performance across a wide range of tasks, making them potentially more suitable alternatives.
3. One point of concern is whether the problems addressed in this paper inherently have strong temporal characteristics. The authors claim that traditional methods fail to perform well due to their inability to consider time-series features. However, evidence and detailed explanations for this claim are lacking. Moreover, the experiments are conducted on a well-established Multi-agent Particle Environment (MPE), and the main contributions of the paper appear to be minor modifications to the neural network architecture and the MADDPG learning approach. It's unclear what distinct advantages these changes offer over existing methods.


### Paper Writing
1. The quality of the paper's writing leaves room for improvement, particularly in reference to Figures 3.1 and 3.2, where the arrows and symbols lack detailed explanations.
2. In this paper, the employed formulaic notation frequently deviates from conventional standards and exhibits a lack of clarity. For instance, in Equation (1), the author neglects to provide a lucid definition of the variable \theta and omits any explanation for introducing the Deterministic Policy Gradient (DPG) method. This issue persists throughout the paper, as numerous formulas are presented without adequate explanation and contextual information.
3. Additionally, the abbreviations in the introduction need to be standardized. Some terms are presented in abbreviated form without explanation. There are also parts where variables are not described, such as the variable τ in the Background section. Is τ an action or a trajectory? The meaning of this variable is clarified later in the paper, but it would be better to mention it in the Background section.

## Possible thoughts
1. Warm start for the process, either teacher-student or demonstration learing(hard to get)
2. chagne the reward func






## Oct 10th:
### Done:
1. Get the environment ready
### Todo:
1. Training task:
   1. simple-tag:MADDPG,MADDPG-LSTM,FAC,IDDPG
   2. simple-spread:MADDPG,MADDPG-LSTM,FAC,IDDPG
   3. simple-spread-6a:MADDPG,MADDPG-LSTM,FAC,IDDPG

## Oct 9th:
### Done:
1. figure out the actual cause of the performance gap.

### Todo:
1. find optimization on the training pipeline.


## Oct 4th:
### Todo:
1. Figure out the performance gap, Current situation is: Performance drop after first batch. Potential reasons are as following:
   1. Maybe only the first batch is running on GPU.(Witnessing no GPU usage rise after first batch; the performance gap happens to fit the batchsize)
   2. Memory limit?(Personally I don't think that's the case)
   3. Torch inherent property? If that's the case then maybe just live with it.
2. Try the transformer-based method that fuses the temporal message.
3. The "max-episode-len" argument is something that we can change for different scenarios.
4. The "select_action" please make it better and easier to add new algorithms.