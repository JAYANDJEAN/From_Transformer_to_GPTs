import os
import gymnasium as gym
from ppo import PPO
import yaml


def train():
    print("=" * 70)
    with open('../00_assets/yml/ppo_trainer.yml', 'r') as file:
        config = yaml.safe_load(file)

    env = gym.make(config['env_name'], hardcore=True)
    print_freq = config['max_ep_len'] * 10  # print avg reward in the interval (in num timesteps)
    log_freq = config['max_ep_len'] * 2  # log avg reward in the interval (in num timesteps)
    update_timestep = config['max_ep_len'] * 4  # update policy every n timesteps

    state_dim = env.observation_space.shape[0]
    # action space dimension
    if config['has_continuous_action_space']:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_dir = log_dir + '/' + config['env_name'] + '/'
    os.makedirs(log_dir, exist_ok=True)
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)
    log_f_name = log_dir + '/PPO_' + config['env_name'] + "_log_" + str(run_num) + ".csv"
    print("current logging run number for " + config['env_name'] + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    run_num_pretrained = 0  # change this to prevent overwriting weights in same env_name folder

    directory = "PPO_PreTrained"
    os.makedirs(directory, exist_ok=True)
    directory = directory + '/' + config['env_name'] + '/'
    os.makedirs(directory, exist_ok=True)
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(config['env_name'],
                                                            config['random_seed'],
                                                            run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    print("=" * 70)
    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim,
                    config['lr_actor'], config['lr_critic'], config['gamma'], config['K_epochs'],
                    config['eps_clip'], config['has_continuous_action_space'], config['action_std'])
    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    print_running_reward = 0
    print_running_episodes = 0
    log_running_reward = 0
    log_running_episodes = 0
    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= config['max_training_timesteps']:
        state = env.reset()
        state = state[0]

        current_ep_reward = 0

        for t in range(1, config['max_ep_len'] + 1):

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, truncated, info = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if config['has_continuous_action_space'] and time_step % config['action_std_decay_freq'] == 0:
                ppo_agent.decay_action_std(config['action_std_decay_rate'], config['min_action_std'])

            # log in logging file
            if time_step % log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                        print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % config['save_model_freq'] == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()


if __name__ == '__main__':
    train()
