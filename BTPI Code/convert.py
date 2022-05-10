import os
import matplotlib.pyplot as plt

# dir = 'Result2_images/old'
# files = os.listdir(dir)

# for fi in files:
#     img = plt.imread(os.path.join(dir, fi))
#     plt.imsave(os.path.join(dir, fi.split('.')[0]+'.png'), img)

epo = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
iou = [0.2523, 0.3031, 0.3383, 0.3962, 0.5276, 0.5191, 0.5550, 0.5339, 0.5974, 0.6780]
diceloss = [0.5462, 0.5662, 0.5567, 0.5220, 0.4624, 0.4681, 0.4410, 0.4500, 0.4160, 0.3784]

plt.figure()
plt.plot(epo, iou)
plt.plot(epo, diceloss)
plt.legend(["IoU Score", "Dice Loss"])
plt.xlabel("Dataset Size")
plt.ylabel("Value")
plt.grid(True)
plt.title("Variations of IoU score and Dice Loss with Dataset Size")
plt.savefig('variantions.png')