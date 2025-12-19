import numpy as np
import gymnasium as gym


env = gym.make('CartPole-v0')
state = env.reset()
done = False

while not done:
    env.render()
    action = np.random.choice([0, 1])
    '''
    nv.step の戻り値は以下
    (
    observation,      # 状態（NumPy 配列）
    reward,           # 報酬
    terminated,       # 終了（成功・失敗などの論理的終了）
    truncated,        # 打ち切り（時間制限など）
    info              # 補足情報
    )
    '''
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(next_state)

env.close()