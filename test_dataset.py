from datasets.davis import DAVISLoader

def test():
    import argparse
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--imgset")
    parser.add_argument("--img_folder")
    args = parser.parse_args()

    dataset = DAVISLoader(root_path=args.root, imageset_folder=args.imgset, image_folder=args.img_folder)
    dataset.__getitem__(0)
    #support_fr, query_img = dataset.__getitem__(0)

    '''
    plt.subplot(1, 3, 1)
    plt.imshow(support_fr[0])
    plt.subplot(1, 3, 2)
    plt.imshow(support_fr[1])
    plt.subplot(1, 3, 3)
    plt.imshow(query_img)
    plt.show()
    plt.close()
    '''

if __name__ == "__main__":
    test()
    