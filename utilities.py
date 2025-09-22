def visualize(x,y1,y2):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,5))
    plt.plot(x, y1, label='train_err')
    plt.plot(x, y2, label='val_err')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

def bar_plot(x,y1,y2):
    import matplotlib.pyplot as plt
    import numpy as np
    features = np.array(x)

    plt.bar(features, y1, color='b', width=0.25)
    plt.bar(features + 0.25, y2, color='g', width=0.25)
    plt.xlabel('Feature IDs', fontweight='bold', fontsize=10)
    plt.ylabel('R2_score', fontweight='bold', fontsize=10)
    plt.xticks(features)
    plt.legend(labels=['Train', 'Val'])
    plt.title('Single Feature Performance - Bar Plot')
    plt.show()

def plot(x,y):
    import matplotlib.pyplot as plt
    
    plt.title(f'log10(Aphas)')
    plt.xticks(x)
    plt.plot(x, y, label='train error')
    plt.grid()
    plt.show()