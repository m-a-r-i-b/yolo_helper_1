import os
from random import shuffle
import sys
def create_split(split, img_dir):
    images = [x for x in os.listdir(img_dir) if not x.endswith('.txt')]
    
    shuffle(images)
    train_size = int(len(images)*split)
    test_size = len(images) - train_size
    val_size = test_size//2
    train_data = images[:train_size]
    test_data = images[train_size:train_size+val_size]
    val_data = images[train_size+val_size:]
    print(f"Train size samples: {train_size}") 
    print(f"Test size  samples: {test_size}") 
    print(f"validation  size samples: {val_size}") 

    write_file('train',train_data, img_dir)
    write_file('test',test_data, img_dir)
    write_file('valid',val_data, img_dir)

def write_file(split_type,images, img_dir):
    with open(split_type+'.txt','w') as fp:
        for image in images:
            fp.write(f'data/{img_dir}/{image}\n')
        


def main():
    split = float(sys.argv[1])
    img_dir = sys.argv[2]
    create_split(split,img_dir)

if __name__ == "__main__":
    main()