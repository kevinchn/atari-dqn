# run learned optimal policy (without exploration)

import gym
import numpy as np
import tensorflow as tf

from dqn import *

ENV_NAME = 'PongNoFrameskip-v4'

def run_dqn(env):
    replay = ReplayMemory(memory_size=10)
    
    with tf.Session() as session:
        saver = tf.train.import_meta_graph('model/model.ckpt.meta')
        saver.restore(session, tf.train.latest_checkpoint('./model'))
        
        obs = env.reset()
        
        while True:
            replay.store_frame(obs)
            frame_stack = replay.get_current_frame_stack()
            actions_q = session.run('q/fully_connected_1/BiasAdd:0', feed_dict={'Placeholder:0': frame_stack})
            action = np.argmax(actions_q)
            
            obs, reward, done, info = env.step(action)
            replay.store_transition(action, reward, done)
            
            if done:
                break

def main():
    will_record_video = lambda episode_id: True
    
    env = gym.make(ENV_NAME)
    env = gym.wrappers.Monitor(env, MONITOR_DIR, video_callable=will_record_video, force=True)
    env = PreprocessFrame(env)
    
    run_dqn(env)

if __name__ == "__main__":
    main()
