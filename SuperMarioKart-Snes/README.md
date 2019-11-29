# Integrating a new game

Install :
* Integration tool
* gym retro

## Downloading the rom

The available formats are :
* .md: Sega Genesis (also known as Mega Drive)
* .sfc: Super Nintendo Entertainment System (also known as Super Famicom)
* .nes: Nintendo Entertainment System (also known as Famicom)
* .a26: Atari 2600
* .gb: Nintendo Game Boy
* .gba: Nintendo Game Boy Advance
* .gbc: Nintendo Game Boy Color
* .gg: Sega Game Gear
* .pce: NEC TurboGrafx-16 (also known as PC Engine)
* .sms: Sega Master System

First check whether or note your game is already in the gym retro evironnement either looking in the module's folders, or just running this scipt :
```bash
python3 -m retro.import /path/to/your/ROMs/directory/
```

To list all the games on your system :
```python
import retro
print(retro.data.list_games())
```


## Inspecting ram

Recommand using Bizhawk, we features a very complete ram search function, and allows to save intermediate results. My analysis on Mario Kart ended in :

| Name | Address | Type | Infos |
| :- | :- | :- | :- |
| |   
| x | 7E0088 | <u2 | Origin is top-left (EAST) |   
| y | 7E008C | <u2 | // (SOUTH) |   
| speed_x | 7e1022 | <i2 | speed_east |   
| speed_y | 7e1024 | <i2 | speed_south |   
| speed | 7e10ea | <u2 | 0-1000, increases with coins |   
| camera_angle | 7E0095 | \|u1 | N0, E64, S128, S192 |   
| angle_velocity | 7e109e | \|u1 | -7 to 7 |   
| hop | 7e101f | <u2 | 0 = on ground, >0 = hop height |   
| kart_angle | 7e10aa | <i2 | Check range value |   
| |   
| checkpoint | 7e10c0 | \|u1 | Does not start at 0, and can decrease through time |   
| checkpoint_sum | 7e0148 | \|u1 | Number of checkpoints |   
| lap | 7e10c1 | \|u1 | At end = 133 |   
| lap_max | 7e10f9 | \|u1 |  |   
|  |   
| milliseconds | 7e0101 | \|d1 |  |   
| seconds | 7e0102 | \|d1 |  |   
| minutes | 7e0104 | \|u1 | test |   
|  |   
| collision | 7e1052 | \|u1 | 7 -> 0 |   
| ground_status | 7e10a0 | \|u1 | on the ground = 0 jump/hop/ramp = 2 fallen off edge = 4 in lava = 6 in deep water = 8 |   
| ground_type | 7e10ae | \|u1 | unused power up square = 20 deep water = 34 mario circuit road / ghost valley road / used power up square / rainbow road = 64 bowser castle = 68 doughnut plains track = 70 koopa beach sand = 74 choco island track = 76 ghost valley |   
| out_of_control | 7e10a6 | \|u1 | normal = 0/2 using a star = 0/2/28 hit a banana-peel = 12 skidded out right = 14 skidded out left = 16 in a mini-boost = 18 at mini-boost peak = 28 hit something (eg a wall) = 18/28 |   
|  |   
| lives | 7e0154 | \|u1 | nb lives +1 |   
| coins | 7e0e00 | \|u1 | Number of coins |   
| rank | 7e1040 | \|u1 | value = (rank-1)*2 |   
|  |   
| lakitu | 00010b | <i2 | 0=None, 4=Back on track, 16=reverse, 64=end |   
|  |   
| item_1 | 7e0d70 | \|u1 | nothing = (0,60) mushroom = (0,52) feather = (1,52) star = (2,56) banana = (3,56) green shell = (4,48)  |   
| item_2 | 7e0c69 | \|u1 | red shell = (5,52) coin = (7,56) lightning = (8,56) selecting/spinning = (0-8,?) |   
| square_item | 7e0d7c | \|u1 |  |  
|  |
| mushroom_boost | 7e104e | \|u1 | true=56, false=26 |  
|  |
| shrunken | 7e1030 |\|u1 |  |   
|  |

Note: You might have to adapt some types as the gym retro environnement gives a lot more options.

## Integrating the game

Open the integration tool
* Add the variables to data.json
* Add scenario to scenario.json

I recommand that you link the integration directory to the file tracked by git (should be smth like retro/data/stable/your_game)
* On windows : 
```bash
mklink /D {file-name} {link-name} 
```
* On linux : 
```bash
ln -s {file-name} {link-name}
```

## Installing the baselines

```bash
git clone https://github.com/openai/baselines/
pip install -e .
```

And now you can try and run the ppo (this can help to have a first idea about what the reward should be like)
```bash
python3 -m retro.examples.ppo --game MarioKartNes
```