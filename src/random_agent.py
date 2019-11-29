import retro
import numpy as np
import time
import cv2

from env import make_env

def main():
    env = make_env(game='SuperMarioKart-Snes', state="RainbowRoad.Act1")
    env.record_movie("test.bk2")
    obs = env.reset()
    while True:
        action = [False]*12
        action[0]=True
        obs, reward, done, info = env.step(0)
        obs = obs[0]*256
        cv2.imwrite("test.jpg", obs)
        time.sleep(.01)
        if reward != 0 :
            print(reward)
        # env.render()
        if done:
            print(info["seconds"])
            length = episode_duration = info["milliseconds"] + (info["seconds"] + info["minutes"]*60)*60
            print(length)
            obs = env.reset()
            break
    env.stop_record()
    env.close()

if __name__ == "__main__":
    main()

    cap.release()
    cv2.destroyAllWindows()
