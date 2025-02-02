from email.mime import image
import imageio
import os

path = 'astar' #'dijkstra'

rgbs=[]
fs = os.listdir(path)
fs.sort(key=lambda x:int(x.split('.')[0]))
for d in fs:
    p = os.path.join(path, d)
    rgbs.append(imageio.imread(p))

imageio.mimsave('{}.gif'.format(path), rgbs, 'GIF', duration=0.5)