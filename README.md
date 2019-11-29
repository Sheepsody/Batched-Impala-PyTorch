# Reinforced learning on マリオカート

This project is still in progress.

It aims to train an IA to play Mario Kart, using deep neural networks and reinforced learning.

I'm still investigating techniques to train efficiently, but the main methods that I chose are :
* Using V-trace to compute the advantage function (cf IMPALA)
* GPU-based batching approach (GA3C), but A3C is sample inefficient so that's why I switched to V-Trace
* Use a replay-memory


Credits :
* Maps : http://www.mariouniverse.com/maps-snes-smk/