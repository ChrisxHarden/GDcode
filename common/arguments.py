import argparse

"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="simple_tag", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")#Maybe we can change it here
    parser.add_argument("--exploration-steps",type=int,default=100000,help="letting the agents move stochastically to have some experience")
    parser.add_argument("--time-steps", type=int, default=2000000, help="number of time steps")
    # 一个地图最多env.n个agents，用户可以定义min(env.n,num-adversaries)个敌人
    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")

    #算法参数
    parser.add_argument("--algorithm", type=str, default="MADDPG", help="Which MARL algorithm to choose")
    parser.add_argument("--share_param", type=bool, default=False, help="Whether to share params across agents")
    parser.add_argument("--share_agent", type=bool, default=False, help="Whether to share agents")
    parser.add_argument("--agent_type", type=str, default="mlp", help="type of the agent")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64,help="the dim of agent fc or rnn network")
    parser.add_argument("--mixer", type=str, default="qmix", help="name of the scenario script")
    parser.add_argument("--agent_return_logits", type=bool, default=False, help="whether agent to return logits")
    parser.add_argument("--mixing_embed_dim", type=int, default=64, help="the dim of mixing embed network")
    parser.add_argument("--q_embed_dim", type=int, default=1, help="the dim of mixing q")
    parser.add_argument("--hypernet_layers", type=int, default=1, help="the number of layers in the hypernet of mixer")
    parser.add_argument("--hypernet_embed", type=int, default=64, help="the dim of mixing hypernet")
    parser.add_argument("--gated", type=bool, default=False, help="whether to gate the q_tot")

    parser.add_argument("--skip_connections", type=bool, default=False, help="whether to use res in mixer")


    # Core training parameters
    parser.add_argument("--lr-actor", type=float, default=1e-4, help="learning rate of actor") #固定的lr?
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.001, help="parameter for updating the target network")#衡量target_network的更新幅度,越大则更倾向于改变?为什么这么做?
    parser.add_argument("--buffer-size", type=int, default=int(5e5), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=16, help="number of episodes to optimize at the same time")
    parser.add_argument("--lstm", type=bool, default=False, help="enable lstm?")
    parser.add_argument("--seq-length", type=int, default=5, help="length of lstm input slice")

    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=10000, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")

    # Evaluate
    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=100, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")
    args = parser.parse_args()

    return args
