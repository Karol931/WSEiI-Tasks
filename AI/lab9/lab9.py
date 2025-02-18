import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

def create_model(env_id="BipedalWalker-v3", num_cpu=4, total_timesteps=1_000_000, save_path="bipedal_walker_ppo"):
    """Creates, trains, and saves a PPO model."""
    vec_env = make_vec_env(env_id, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)
    
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log="./bipedal_walker_tensorboard/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        max_grad_norm=0.5,
    )
    
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=300, verbose=1)
    eval_callback = EvalCallback(
        vec_env,
        callback_on_new_best=callback_on_best,
        eval_freq=1000,
        verbose=1,
    )
    
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(save_path)
    vec_env.close()

def test_model(model_path="bipedal_walker_ppo", env_id="BipedalWalker-v3", render=True):
    """Loads a trained model and tests it in a Gym environment."""
    env = gym.make(env_id, render_mode="human")
    model = PPO.load(model_path)
    
    obs, _ = env.reset()
    total_reward = 0
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        if render:
            env.render()
        
        if terminated or truncated:
            break
    
    print(f"Total reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    create_model()
    test_model()
