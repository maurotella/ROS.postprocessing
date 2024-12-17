import stats
import matplotlib.pyplot as plt
import sqlite3, zlib, pickle
import numpy as np
import importlib

fig, ax = plt.subplots(dpi=200)

conn = sqlite3.connect('normRawMaps.db')
MAPS = conn.execute('SELECT * FROM RawMaps').fetchall()
MAPS = {m[0]:pickle.loads(zlib.decompress(m[1])) for m in MAPS}
MAPS.keys()
MAP = MAPS['office_2']
res = MAP['info']['resolution']
o_x = MAP['info']['origin']['position']['x']
o_y = MAP['info']['origin']['position']['y']
height, width = MAP['info']['height'], MAP['info']['width']
npMaps = stats.get_map_img(MAP)
print(MAPS.keys())
plt.ion()
stats.plot_map(npMaps,ax)
plt.xticks(
    [x for x in np.arange(0,width,200)],
    [round(x*res+o_x) for x in np.arange(0,width,200)]
)
plt.yticks(
    [y for y in np.arange(0,height,200)],
    [round(y*res+o_y) for y in np.arange(0,height,200)]
)
colors = ['orange','red','blue','black','yellow','purple']
point_id = 0
points = [[]]
def onclick(event):
    if event.button==1: #tasto sx mouse
        x,y = event.xdata,event.ydata
        points[point_id].append((x,y))
        if len(points[point_id])!=1:
            ax.plot(
                [points[point_id][-1][0],points[point_id][-2][0]],
                [points[point_id][-1][1],points[point_id][-2][1]],
                color='black',
                zorder=1
            )
        ax.plot(points[point_id][-1][0],points[point_id][-1][1],'o',color=colors[point_id%6],markersize=3,zorder=2)
            
def options(event):
    global point_id, points
    if event.key=='n':
        ax.plot(
            [points[point_id][-1][0],points[point_id][0][0]],
            [points[point_id][-1][1],points[point_id][0][1]],
            color='black',
            zorder=1
        )
        point_id += 1
        points.append([])
    if event.key=='d':
        points[-1] = points[-1][:-1]
        ax.cla()
        stats.plot_map(npMaps,ax)
        for pid, p in enumerate(points):
            for idx, point in enumerate(p):
                ax.plot(
                    *point,
                    'o',
                    color=colors[pid%6],
                    markersize=3,
                    zorder=2
                )
                if idx!=len(p)-1:
                    ax.plot(
                        [point[0],p[idx+1][0]],
                        [point[1],p[idx+1][1]],
                        color='black',
                        zorder=1
                    )
                elif pid!=len(points)-1:
                    ax.plot(
                        [point[0],p[0][0]],
                        [point[1],p[0][1]],
                        color='black',
                        zorder=1
                    )


def close(_):
    np_points = [
        [tuple(np.round((x*res+o_x, y*res+o_y),2)) for x,y in p]
        for p in points[:-1]
    ]
    n = 1
    for p in np_points:
        print(f'Area {n}:',p)
        n += 1
    #print([np.round(np.array(p)*res,2) for p in points])

fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', options)
fig.canvas.mpl_connect('close_event', close)
plt.pause(0.01)
plt.show(block=True)

