import numpy as np
import matplotlib.pyplot as plt

#PALETTE = ["8ecae6","219ebc","023047","ffb703","fb8500"]
#PALETTE = ["f4f1de","e07a5f","3d405b","81b29a","f2cc8f"]
#PALETTE = ["390099","9e0059","ff0054","ff5400","ffbd00"]
PALETTE = ["001427","708d81","f4d58d","bf0603","8d0801"] # this one
#PALETTE = ["3d348b","7678ed","f7b801","f18701","f35b04"]
#PALETTE = ["606c38","283618","fefae0","dda15e","bc6c25"]

FIG_SIZE = (6, 3)


def create_palette(number_of_colors: int, normalize=True):
    def _create_color(value) -> tuple:
        value = max(0, min(1, value))
        num_colors = len(PALETTE)

        index = value * (num_colors - 1)

        index_low = max(0, min(num_colors - 1, int(np.floor(index))))
        index_high = max(0, min(num_colors - 1, int(np.ceil(index))))

        color_low = PALETTE[index_low]
        color_high = PALETTE[index_high]

        if isinstance(color_low, str):
            color_low = tuple(int(color_low[i:i+2], 16) for i in (0, 2, 4))
            color_high = tuple(int(color_high[i:i+2], 16) for i in (0, 2, 4))   

        t = index - index_low
        interpolated_color = tuple(int((1 - t) * c1 + t * c2) for c1, c2 in zip(color_low, color_high))
        
        if normalize:
            interpolated_color = list(c/255 for c in interpolated_color)
            interpolated_color.extend([1])

        return interpolated_color

    values = np.linspace(0, 1, number_of_colors)
    return [_create_color(val) for val in values]

def show_palette(palette:list, save = False, name = 'palette.png'):
    plt.figure(figsize=(8, 2))
    plt.imshow([palette], aspect='auto', extent=[0, 1, 0, 1])
    plt.title('Pallette')
    plt.xticks([])
    plt.yticks([])

    if save:
        plt.savefig(name, bbox_inches='tight')
    else:
        plt.show()    
    plt.close()