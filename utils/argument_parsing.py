import argparse

def parse_all_arguments():
        
    parser = argparse.ArgumentParser() #description='Running {}'.format(algo_name))
    
    # Basic agruments 
    parser.add_argument('--algo-name', default="TRPO", metavar='G',
                       help='algorithm name')
    parser.add_argument('--model-path', metavar='G',
                        help='path of pre-trained model')
    parser.add_argument('--exp-num', default="1", metavar='G',
                        help='Experiment number for today (default: 1)')
    parser.add_argument('--exp-name', default="Exp-1", metavar='G',
                        help='Experiment name')
    parser.add_argument('--env-name', default="CartPole-v0", metavar='G',
                        help='name of the environment to run')
    
    # Learning rates and regularizations
    parser.add_argument('--render', action='store_true', default=False,
                        help='render the environment')
    parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                        help='log std for the policy (default: -0.0)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                        help='gae (default: 0.95)')
    parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                        help='l2 regularization of value function (default: 1e-3)')
    parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='G',
                        help='gae (default: 3e-4)')
    
    # GPU index, multi-threading and seeding
    parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
    parser.add_argument('--num-threads', type=int, default=3, metavar='N',
                        help='number of threads for agent (default: 4)')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random seed (default: 1)')
    
    # batch size and iteration number
    parser.add_argument('--min-batch-size', type=int, default=2000, metavar='N',
                        help='minimal batch size per PPO update (default: 2048)')
    parser.add_argument('--max-batch-size', type=int, default=2000, metavar='N',
                        help='maximum batch size per PPO update (default: 2000)')
    parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                        help='maximal number of main iterations (default: 500)')
    
    # logging and saving models
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--save-model-interval', type=int, default=50, metavar='N',
                        help="interval between saving model (default: 0, means don't save)")
    parser.add_argument('--save-intermediate-model', type=int, default=10, metavar='N',
                        help="intermediate model saving interval (default: 0, means don't save)")
       
    
    if parser.parse_args().algo_name == "CPO":
        parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                        help='max kl value (default: 1e-2)')
        parser.add_argument('--damping', type=float, default=1e-2, metavar='G',
                        help='damping (default: 1e-2)')
        parser.add_argument('--max-constraint', type=float, default=1e-3, metavar='G',
                        help='max constraint value (default: 1e-2)')
        parser.add_argument('--annealing_factor', type=float, default=1e-6, metavar='G',
                        help='annealing factor of constraint (default: 1e-2)')
        parser.add_argument('--anneal', default=True,
                        help='Should the constraint be annealed or not')
        parser.add_argument('--grad-norm', default=False,
                        help='Should the norm of policy gradient be taken (default: False)')    
    elif parser.parse_args().algo_name == "TRPO":
        parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                        help='max kl value (default: 1e-2)')
        parser.add_argument('--damping', type=float, default=1e-2, metavar='G',
                        help='damping (default: 1e-2)')
    
    return parser.parse_args()

