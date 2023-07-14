import matplotlib.pyplot as plt
import seaborn

def acc_and_mac_f1_image_only():
    """ACC and MAC_F1(image only)"""
    acc = [62, 63.75, 63.5, 65.25, 64.5, 62.5, 63, 61.5, 64.75, 63]
    mac_f1 = [34.49, 52.54, 54.24, 55.66, 48.95, 50.09, 55.27, 51.93, 55.82, 54.48]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = range(1, 11)
    
    ax.plot(x, acc, label='ACC', marker='o')
    ax.plot(x, mac_f1, label='MAC_F1', marker='o')
    ax.set_xlabel('EPOCHS')
    ax.set_ylabel('SCORE')
    ax.set_xticks(x)
    ax.set_ylim((0, 100))
    ax.set_title('ACC and MAC_F1 on validation set(image only)')
    plt.legend()
    plt.show()



