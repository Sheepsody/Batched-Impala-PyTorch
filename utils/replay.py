import retro
import argparse
import os


parser = argparse.ArgumentParser()

parser.add_argument("-m", "--movie", help="Movie to be replayed", required=True)


def replay_movie(movie_path):

    movie = retro.Movie(movie_path)
    movie.step()

    env = retro.make(game=movie.get_game(), state=retro.State.NONE, use_restricted_actions=retro.Actions.ALL)
    env.initial_state = movie.get_state()
    env.reset()

    print('Stepping movie')

    while movie.step():
        keys = []
        for i in range(len(env.buttons)):
            keys.append(movie.get_key(i, 0))
        _obs, _rew, _done, _info = env.step(keys)
        env.render()
    
    env.close()


if __name__ == "__main__":
    args = parser.parse_args()

    assert os.path.exists(args.movie), "No movie could be found"

    replay_movie(args.movie)
