import os.path
import shutil


def main():
    orgimg_dir = 'data/celeba_hq/train'
    reorderimg_dir = 'evaldata/celeba_hq/train'
    orglabels = ['Male', 'Smiling']
    labels = []
    for label in orglabels:
        labels.append(label)
        labels.append("Not_" + label)

    annof = open("expr/CelebAMask-HQ-attribute-anno.txt", "r")
    mapf = open("expr/CelebA-HQ-to-CelebA-mapping.txt", "r")

    num = annof.readline()
    label_header = annof.readline().split()
    for label in orglabels:
        assert label in label_header
    labels_idx = [label_header.index(label) for label in orglabels]

    # imglist = [None] * (2 ^ len(orglabels))
    imglist = [[] for _ in range(4)]
    for line in annof.readlines():
        line = line.split()
        file_name, tmplabel = line[0], line[1:]

        if tmplabel[labels_idx[0]] == "1":
            if tmplabel[labels_idx[1]] == "1":
                imglist[0].append(file_name)
            else:
                imglist[1].append(file_name)
        else:
            if tmplabel[labels_idx[1]] == "1":
                imglist[2].append(file_name)
            else:
                imglist[3].append(file_name)

    header = mapf.readline()
    lines = mapf.readlines()

    annof.close()
    mapf.close()

    dirlist = os.listdir(orgimg_dir)
    for i in range(4):
        if i == 0:
            newsubdir = orglabels[0] + "-" + orglabels[1]
        elif i == 1:
            newsubdir = orglabels[0] + '-Not_' + orglabels[1]
        elif i == 2:
            newsubdir = "Not_" + orglabels[0] + "-" + orglabels[1]
        else:
            newsubdir = "Not_" + orglabels[0] + '-Not_' + orglabels[1]

        for fileidx in imglist[i]:
            idx = int(fileidx.split('.')[0])
            filename = lines[idx].split()[-1]

            for subdir in dirlist:
                tmppath = os.path.join(orgimg_dir, subdir, filename)
                if os.path.exists(tmppath):
                    newpath = os.path.join(reorderimg_dir, newsubdir)
                    os.makedirs(newpath, exist_ok=True)
                    shutil.copy(tmppath, os.path.join(newpath, filename))
                    break

if __name__ == "__main__":
    main()