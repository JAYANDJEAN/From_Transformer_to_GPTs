import os
import gymnasium as gym
from ppo import PPO
import yaml
import glob
from PIL import Image

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
    env = gym.make(config['env_name'])
    env.metadata["render_modes"] = 'rgb_array'
    total_test_episodes = 1
    total_timesteps = 300
    step = 10
    frame_duration = 150

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
    test_running_reward = 0

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
        ep_reward = 0

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
    img.save(fp=gif_path, format='GIF', append_images=imgs, save_all=True, optimize=True, duration=frame_duration,
             loop=0)
    print("saved gif at : ", gif_path)


if __name__ == '__main__':
    train()
