from stable_baselines3 import A2C, DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from tqdm import tqdm
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy

env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=16, seed=0)
env = VecFrameStack(env, n_stack=4)
tmp_path = "logs"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

model = A2C("CnnPolicy", env, verbose=1)
model.set_logger(new_logger)

try:
    for i in tqdm(range(100)):
        model.learn(total_timesteps=100_000, progress_bar=True)
        model.save(f"model/model_{i}")
except Exception as e:
    print(e)
finally:
    model.save(f"model/model_final")

reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

