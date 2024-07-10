import os
import gymnasium as gym
from ppo import PPO
import yaml
import glob
from PIL import Image

import pandas as pd
import matplotlib.pyplot as plt

with open('../00_assets/yml/ppo_trainer.yml', 'r') as file:
    config = yaml.safe_load(file)


def train():
    print("=" * 90)
    env = gym.make(config['env_name'], hardcore=True)

    #####################################################
    log_dir = config['log_dir'] + '/' + config['env_name'] + '/'
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    cur_file_num = len(next(os.walk(log_dir))[2])
    log_f_name = log_dir + f"log_{config['env_name'].lower()}_{cur_file_num}.csv"
    print("logging at : " + log_f_name)

    model_dir = config['model_dir'] + '/' + config['env_name'] + '/'
    os.makedirs(config['model_dir'], exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    checkpoint_path = model_dir + f"model_{config['env_name'].lower()}_{config['random_seed']}.pth"
    print("save model at : " + checkpoint_path)
    print("=" * 90)

    # initialize a PPO agent
    state_dim = env.observation_space.shape[0]
    if config['has_continuous_action_space']:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    ppo_agent = PPO(state_dim, action_dim,
                    config['lr_actor'], config['lr_critic'], config['gamma'], config['K_epochs'],
                    config['eps_clip'], config['has_continuous_action_space'], config['action_std'])
    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')

    #####################################################
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
            action = ppo_agent.select_action(state)
            state, reward, done, truncated, info = env.step(action)
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            if time_step % (config['max_ep_len'] * 4) == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if config['has_continuous_action_space'] and time_step % config['action_std_decay_freq'] == 0:
                ppo_agent.decay_action_std(config['action_std_decay_rate'], config['min_action_std'])

            # log in logging file
            if time_step % (config['max_ep_len'] * 2) == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)
                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()
                log_running_reward = 0
                log_running_episodes = 0

            if time_step % (config['max_ep_len'] * 10) == 0:
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)
                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                        print_avg_reward))
                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % config['save_model_freq'] == 0:
                print("-" * 80)
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("-" * 80)

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


def visual():
    print("=" * 90)
    env = gym.make(config['env_name'], render_mode='rgb_array')
    total_test_episodes = 1
    total_timesteps = 300
    step = 10
    frame_duration = 150
    test_running_reward = 0

    image_dir = config['image_dir'] + '/' + config['env_name'] + '/'
    gif_dir = config['gif_dir'] + '/' + config['env_name'] + '/'
    model_dir = config['model_dir'] + '/' + config['env_name'] + '/'

    os.makedirs(config['image_dir'], exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(config['gif_dir'], exist_ok=True)
    os.makedirs(gif_dir, exist_ok=True)

    gif_path = gif_dir + f"gif_{config['env_name'].lower()}_{config['random_seed']}.gif"
    checkpoint_path = model_dir + f"model_{config['env_name'].lower()}_{config['random_seed']}.pth"
    print("loading network from : " + checkpoint_path)

    state_dim = env.observation_space.shape[0]
    if config['has_continuous_action_space']:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    ppo_agent = PPO(state_dim, action_dim,
                    config['lr_actor'], config['lr_critic'], config['gamma'], config['K_epochs'],
                    config['eps_clip'], config['has_continuous_action_space'], config['action_std'])
    ppo_agent.load(checkpoint_path)

    print("-" * 80)

    for ep in range(1, total_test_episodes + 1):
        ep_reward = 0
        state = env.reset()
        state = state[0]
        for t in range(1, config['max_ep_len'] + 1):
            action = ppo_agent.select_action(state)
            state, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            img = env.render()
            img = Image.fromarray(img)
            img.save(image_dir + str(t).zfill(6) + '.jpg')
            if done:
                break
        ppo_agent.buffer.clear()
        test_running_reward += ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
    env.close()

    print("=" * 90)
    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("=" * 90)
    img_paths = sorted(glob.glob(image_dir + '*.jpg'))
    img_paths = img_paths[:total_timesteps]
    img_paths = img_paths[::step]
    print("total frames in gif : ", len(img_paths))
    print("total duration of gif : " + str(round(len(img_paths) * frame_duration / 1000, 2)) + " seconds")
    img, *imgs = [Image.open(f) for f in img_paths]
    img.save(fp=gif_path, format='GIF', append_images=imgs, save_all=True,
             optimize=True, duration=frame_duration, loop=0)
    print("saved gif at : ", gif_path)


def reward_graph():
    print("=" * 90)
    plot_avg = True  # plot average of all runs; else plot all runs separately
    fig_width = 10
    fig_height = 6

    window_len_smooth = 20
    min_window_len_smooth = 1
    linewidth_smooth = 1.5
    alpha_smooth = 1
    window_len_var = 5
    min_window_len_var = 1
    linewidth_var = 2
    alpha_var = 0.1
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'olive', 'brown',
              'magenta', 'cyan', 'crimson', 'gray', 'black']

    # make directory for saving figures

    figure_dir = config['figure_dir'] + '/' + config['env_name'] + '/'
    log_dir = config['log_dir'] + '/' + config['env_name'] + '/'

    os.makedirs(config['figure_dir'], exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)
    fig_save_path = figure_dir + f"fig_{config['env_name'].lower()}_{config['random_seed']}.png"
    current_num_files = next(os.walk(log_dir))[2]
    num_runs = len(current_num_files)
    all_runs = []

    for run_num in range(num_runs):
        log_f_name = log_dir + f"log_{config['env_name'].lower()}_{run_num}.csv"
        print("loading data from : " + log_f_name)
        data = pd.read_csv(log_f_name)
        data = pd.DataFrame(data)
        print("data shape : ", data.shape)
        all_runs.append(data)
        print("-" * 80)

    ax = plt.gca()

    if plot_avg:
        df_concat = pd.concat(all_runs)
        df_concat_groupby = df_concat.groupby(df_concat.index)
        data_avg = df_concat_groupby.mean()
        data_avg['reward_smooth'] = data_avg['reward'].rolling(window=window_len_smooth, win_type='triang',
                                                               min_periods=min_window_len_smooth).mean()
        data_avg['reward_var'] = data_avg['reward'].rolling(window=window_len_var, win_type='triang',
                                                            min_periods=min_window_len_var).mean()

        data_avg.plot(kind='line', x='timestep', y='reward_smooth', ax=ax, color=colors[0], linewidth=linewidth_smooth,
                      alpha=alpha_smooth)
        data_avg.plot(kind='line', x='timestep', y='reward_var', ax=ax, color=colors[0], linewidth=linewidth_var,
                      alpha=alpha_var)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[0]], ["reward_avg_" + str(len(all_runs)) + "_runs"], loc=2)
    else:
        for i, run in enumerate(all_runs):
            # smooth out rewards to get a smooth and a less smooth (var) plot lines
            run['reward_smooth_' + str(i)] = run['reward'].rolling(window=window_len_smooth, win_type='triang',
                                                                   min_periods=min_window_len_smooth).mean()
            run['reward_var_' + str(i)] = run['reward'].rolling(window=window_len_var, win_type='triang',
                                                                min_periods=min_window_len_var).mean()
            run.plot(kind='line', x='timestep', y='reward_smooth_' + str(i), ax=ax, color=colors[i % len(colors)],
                     linewidth=linewidth_smooth, alpha=alpha_smooth)
            run.plot(kind='line', x='timestep', y='reward_var_' + str(i), ax=ax, color=colors[i % len(colors)],
                     linewidth=linewidth_var, alpha=alpha_var)
        handles, labels = ax.get_legend_handles_labels()
        new_handles = []
        new_labels = []
        for i in range(len(handles)):
            if i % 2 == 0:
                new_handles.append(handles[i])
                new_labels.append(labels[i])
        ax.legend(new_handles, new_labels, loc=2)
    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)
    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Rewards", fontsize=12)
    plt.title(config['env_name'], fontsize=14)
    fig = plt.gcf()
    fig.set_size_inches(fig_width, fig_height)
    print("=" * 90)
    plt.savefig(fig_save_path)
    print("figure saved at : ", fig_save_path)
    plt.show()


if __name__ == '__main__':
    visual()
