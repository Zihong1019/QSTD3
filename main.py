import argparse
import os
import numpy as np
import gym
import torch
import utils
import QSTD3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="HalfCheetah-v4", help="Name of the environment")
    parser.add_argument("--seed", default=0, type=int, help="Random seed for reproducibility")
    parser.add_argument("--start_timesteps", default=25e3, type=int, help="Time steps initially purely random policy is used")
    parser.add_argument("--eval_freq", default=5e3, type=int, help="Frequency of evaluations (in time steps)")
    parser.add_argument("--max_timesteps", default=1e6, type=int, help="Total number of time steps to train the agent")
    parser.add_argument("--expl_noise", default=0.1, type=float, help="Standard deviation of Gaussian exploration noise")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size for both actor and critic training")
    parser.add_argument("--discount", default=0.99, type=float, help="Discount factor for future rewards")
    parser.add_argument("--tau", default=0.005, type=float, help="Target network update rate")
    parser.add_argument("--policy_noise", default=0.2, type=float, help="Noise added to target policy during critic update")
    parser.add_argument("--noise_clip", default=0.5, type=float, help="Range to clip target policy noise")
    parser.add_argument("--policy_freq", default=2, type=int, help="Frequency of delayed policy updates")
    parser.add_argument("--save_model", action="store_true", help="Flag to save the model")
    parser.add_argument("--load_model", default="", help="Path to load a pre-trained model")
    parser.add_argument("--k", default=3, type=int, help="Transfer Buffer factor for batch size during sample selection")

    args = parser.parse_args()

    file_name = f"QSTD3_{args.env}_{args.seed}"
    best_model_file = f"./models/{file_name}_best"

    print("---------------------------------------")
    print(f"Policy: QSTD3, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
    }

    policy = QSTD3.QSTD3(**kwargs)
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    evaluations = [utils.eval_policy(policy, args.env, args.seed)]
    best_eval_reward = -np.inf

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            state = np.array(state).reshape(1, -1)
            action = (
                policy.select_action(state)
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        if t >= args.start_timesteps:
            sample_state, sample_action, sample_next_state, sample_reward, sample_not_done = policy.sample_with_value(
                replay_buffer, args.batch_size, args.k)
            policy.train(sample_state, sample_action, sample_next_state, sample_reward, sample_not_done)

        if done:
            print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        if (t + 1) % args.eval_freq == 0:
            eval_reward = utils.eval_policy(policy, args.env, args.seed)
            evaluations.append(eval_reward)
            np.save(f"./results/{file_name}", evaluations)
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                policy.save(best_model_file)
            if args.save_model:
                policy.save(f"./models/{file_name}")

    print(f"Training completed. Best evaluation reward: {best_eval_reward}")