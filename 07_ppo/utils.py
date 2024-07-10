import os
import glob
from PIL import Image
from ppo import PPO
import gymnasium as gym


def save_gif_images(config):
    print("=" * 90)
    env = gym.make(config['env_name'])
    env.metadata["render_modes"] = 'rgb_array'
    total_test_episodes = 1
    gif_num = 0
    total_timesteps = 300
    step = 10
    frame_duration = 150

    gif_images_dir = "PPO_gif_images" + '/'
    os.makedirs(gif_images_dir, exist_ok=True)
    gif_images_dir = gif_images_dir + '/' + config['env_name'] + '/'
    os.makedirs(gif_images_dir, exist_ok=True)
    gif_dir = "PPO_gifs" + '/'
    os.makedirs(gif_dir, exist_ok=True)
    gif_dir = gif_dir + '/' + config['env_name'] + '/'
    os.makedirs(gif_dir, exist_ok=True)

    gif_images_dir = "PPO_gif_images/" + config['env_name'] + '/*.jpg'
    gif_dir = "PPO_gifs"
    os.makedirs(gif_dir, exist_ok=True)
    gif_dir = gif_dir + '/' + config['env_name']
    os.makedirs(gif_dir, exist_ok=True)
    gif_path = gif_dir + '/PPO_' + config['env_name'] + '_gif_' + str(gif_num) + '.gif'

    directory = "PPO_PreTrained" + '/' + config['env_name'] + '/'
    checkpoint_path = directory + "PPO_{}_{}.pth".format(config['env_name'], config['random_seed'])
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
            img.save(gif_images_dir + '/' + str(t).zfill(6) + '.jpg')

            if done:
                break

        ppo_agent.buffer.clear()

        test_running_reward += ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    print("=" * 90)
    print("total number of frames / timesteps / images saved : ", t)
    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("=" * 90)
    img_paths = sorted(glob.glob(gif_images_dir))
    img_paths = img_paths[:total_timesteps]
    img_paths = img_paths[::step]

    print("total frames in gif : ", len(img_paths))
    print("total duration of gif : " + str(round(len(img_paths) * frame_duration / 1000, 2)) + " seconds")
    img, *imgs = [Image.open(f) for f in img_paths]
    img.save(fp=gif_path, format='GIF', append_images=imgs, save_all=True, optimize=True, duration=frame_duration,
             loop=0)
    print("saved gif at : ", gif_path)




