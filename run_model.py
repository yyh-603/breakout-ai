from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.env_util import make_atari_env

env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=1)
env = VecFrameStack(env, n_stack=4)
obs = env.reset()


model = DQN.load(f"model_dqn", env=env, custom_objects={"buffer_size": 10000})
reward, std = evaluate_policy(model, model.env, n_eval_episodes=5, deterministic=False)
print(reward, std)
