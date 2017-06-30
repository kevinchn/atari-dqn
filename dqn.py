from __future__ import print_function

import gym
import numpy as np
import tensorflow as tf
import cv2
import random
import logging

logging.getLogger().setLevel(logging.INFO)

ENV_NAME = 'PongNoFrameskip-v4'
VIDEO_EVERY_N_EPISODES = 100
MONITOR_DIR = './monitoring'
PERF_METRICS_FILE = 'perf_metrics.npy'


class ReplayMemory():
    """Stores past transition experience for training
    to decorrelate samples
    """

    def __init__(self, memory_size=1000000, stack_size=4):
        self.memory_size = memory_size
        self.stack_size = stack_size
        self.frames = None  # shape to be determined by first frame
        self.actions = np.empty(self.memory_size, dtype=np.int32)
        self.rewards = np.empty(self.memory_size, dtype=np.float32)
        self.done = np.empty(self.memory_size, dtype=np.bool)
        self.current_idx = 0
        self.num_data = 0

    def _stack_frame(self, end_idx):
        start_idx = end_idx - self.stack_size + 1
        stack = []
        for idx in range(start_idx, end_idx + 1):
            stack.append(self.frames[idx % self.memory_size])
        return np.stack(stack, axis=-1)

    def store_frame(self, frame):
        if self.frames is None:
            self.frames = np.empty((self.memory_size,) + frame.shape, dtype=np.uint8)
        self.frames[self.current_idx] = frame

    def get_current_frame_stack(self):
        return np.expand_dims(self._stack_frame(self.current_idx), axis=0)

    def store_transition(self, action, reward, done):
        self.actions[self.current_idx] = action
        self.rewards[self.current_idx] = reward
        self.done[self.current_idx] = done
        self.current_idx = (self.current_idx + 1) % self.memory_size
        if self.num_data <= self.memory_size:
            self.num_data += 1

    def sample(self, batch_size):
        if self.num_data < self.memory_size:
            idxes = random.sample(xrange(self.stack_size - 1, self.num_data), batch_size)
        else:
            idxes = random.sample(xrange(self.memory_size), batch_size)
        input_sample = np.stack([self._stack_frame(idx) for idx in idxes])
        action_sample = self.actions[idxes]
        reward_sample = self.rewards[idxes]
        output_sample = np.stack([self._stack_frame(idx + 1) for idx in idxes])
        done_sample = self.done[idxes]

        return input_sample, action_sample, reward_sample, output_sample, done_sample


class PreprocessFrame(gym.Wrapper):
    """Preprocess frame. Take max pixel value over 2 frames to remove flicker.
    Extract Y channel for gray scale. Downsample and crop to 84x84.
    One action steps through 4 frames.
    Clips rewards to -1, 1
    """

    def __init__(self, env=None, frame_skip=4):
        super(PreprocessFrame, self).__init__(env)
        self.prev_frame = None
        self.frame_skip = frame_skip

    def _process_frame(self, frame):
        y_channel = 0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]
        resized = cv2.resize(y_channel, (84, 110), interpolation=cv2.INTER_LINEAR)
        cropped = resized[18:102, :]
        return cropped

    def _step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.frame_skip):
            obs, reward, done, info = self.env.step(action)
            reward = np.sign(reward)
            total_reward += reward
            if done:
                break
            self.prev_frame = obs

        max_frame = np.maximum(self.prev_frame, obs)

        return self._process_frame(max_frame), total_reward, done, info

    def _reset(self):
        obs = self.env.reset()
        self.prev_frame = obs
        return self._process_frame(obs)


def q_network(img, num_actions, scope=None):
    with tf.variable_scope(scope):
        out = img
        out = tf.layers.conv2d(out, filters=32, kernel_size=8, strides=4, padding='same', activation=tf.nn.relu)
        out = tf.layers.conv2d(out, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)
        out = tf.layers.conv2d(out, filters=64, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)
        out = tf.reshape(out, shape=[tf.shape(out)[0], np.prod(out.get_shape().as_list()[1:])])
        out = tf.layers.dense(out, units=512, activation=tf.nn.relu)
        out = tf.layers.dense(out, units=num_actions, activation=None)
        return out


def make_model(num_actions, discount):
    # q network
    # obs here is the frame stack, not the raw observation from environment
    obs = tf.placeholder(tf.uint8, [None, 84, 84, 4])
    q = q_network(tf.cast(obs, tf.float32), num_actions, 'q')  # (?, num_actions)

    # target network
    action = tf.placeholder(tf.int32, [None])
    reward = tf.placeholder(tf.float32, [None])
    next_obs = tf.placeholder(tf.uint8, [None, 84, 84, 4])
    done = tf.placeholder(tf.float32, [None])
    target_q = q_network(tf.cast(next_obs, tf.float32), num_actions, 'target_q')  # (?, num_actions)

    # double dqn - use current q's action instead of target q's action
    action_target_q = (reward + (1 - tf.cast(done, tf.float32)) * discount *
        tf.reduce_sum(tf.one_hot(tf.argmax(q, axis=1), num_actions) * target_q, axis=1))  # (?,)
    action_q = tf.reduce_sum(q * tf.one_hot(action, num_actions), axis=1)  # (?,)
    # huber loss on temporal difference for effectively gradient clipping
    diff = action_target_q - action_q
    #total_error = tf.reduce_mean(tf.where(tf.abs(diff) < 0.5, tf.square(diff), (tf.abs(diff) - 0.25)))
    total_error = tf.reduce_mean(tf.square(diff))

    # training
    q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q')
    train_op = tf.train.AdamOptimizer().minimize(total_error, var_list=q_vars)

    # update target q
    target_q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_q')
    update_target_op = tf.group(*[var_target.assign(var)
                               for var, var_target in zip(sorted(q_vars, key=lambda v: v.name),
                                                          sorted(target_q_vars, key=lambda v: v.name))])

    return {
        'q': q,
        'obs': obs,
        'action': action,
        'reward': reward,
        'next_obs': next_obs,
        'done': done,
        'train_op': train_op,
        'update_target_op': update_target_op
    }

def get_holdout_states(num_states=32):
    # need env without monitor
    env = gym.make(ENV_NAME)
    env = PreprocessFrame(env)
    replay = ReplayMemory()
    obs = env.reset()
    
    num_episodes = 0
    t_step = 0
    while True:
        t_step += 1
        replay.store_frame(obs)
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        replay.store_transition(action, reward, done)
        if done:
            num_episodes += 1
            obs = env.reset()
        if num_episodes >= 10 and t_step >= 10000:
            # sample from at least 10 episodes and 10000 timesteps
            break
    states, _, _, _, _ = replay.sample(num_states)
    return states    

def learn(env, monitor, load_model=True):
    # parameters from Nature paper
    target_network_update_freq = 10000
    discount = 0.99
    update_freq = 4
    replay_start_size = 50000
    
    # for evaluation
    logging.info('generating holdout states')
    holdout_states = get_holdout_states()
    evaluation_metrics = []

    replay = ReplayMemory()

    obs = env.reset()

    def exploration_rate(t_step):
        return max(1 - t_step / 1000000.0, 0) * 0.9 + 0.1

    with tf.Session() as session:
        model = make_model(env.action_space.n, discount)
        session.run(tf.global_variables_initializer())
        session.run(model.get('update_target_op'))

        t_step = 0
        num_param_updates = 0
        logging.info('start')
        while True:
            t_step += 1

            # step through environment
            replay.store_frame(obs)
            # epsilon greedy policy
            if exploration_rate(t_step) > np.random.rand():
                action = env.action_space.sample()
            else:
                frame_stack = replay.get_current_frame_stack()
                actions_q = session.run(model.get('q'), feed_dict={model.get('obs'): frame_stack})
                action = np.argmax(actions_q)
            obs, reward, done, info = env.step(action)
            replay.store_transition(action, reward, done)
            if done:
                obs = env.reset()

            # train q network
            if (t_step > replay_start_size and t_step % update_freq == 0):
                input_sample, action_sample, reward_sample, output_sample, done_sample = replay.sample(32)
                session.run(model.get('train_op'), feed_dict={
                    model.get('obs'): input_sample,
                    model.get('action'): action_sample,
                    model.get('reward'): reward_sample,
                    model.get('next_obs'): output_sample,
                    model.get('done'): done_sample
                })
                num_param_updates += 1
                if num_param_updates % target_network_update_freq == 0:
                    session.run(model.get('update_target_op'))

            # track performance
            if t_step > replay_start_size and t_step % 10000 == 0:
                episode_rewards = monitor.get_episode_rewards()
                average_episode_reward = np.mean(episode_rewards)
                holdout_q = session.run(model.get('q'), feed_dict={model.get('obs'): holdout_states})
                average_max_q = np.mean(np.max(holdout_q, axis=1))
                logging.info("timestep %d" % (t_step,))
                logging.info("episodes %d" % len(episode_rewards))
                logging.info("average episode score %f" % average_episode_reward)
                logging.info("average action value %f" % average_max_q)
                evaluation_metrics.append([t_step, average_episode_reward, average_max_q])
                np.save(PERF_METRICS_FILE, evaluation_metrics)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', action='store_true')
    args = parser.parse_args()

    env = gym.make(ENV_NAME)
    env = monitor = gym.wrappers.Monitor(env, MONITOR_DIR,
                                         video_callable=lambda episode_id: episode_id % VIDEO_EVERY_N_EPISODES == 0,
                                         force=True)
    env = PreprocessFrame(env)
    learn(env, monitor, args.load_model)


if __name__ == "__main__":
    main()
