#!/usr/bin/env python
# -*- coding: utf-8 -*-

from audioop import reverse
import sys
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import random
import re

def get_clusters_and_cost(data, medoids):
    clusters = []
    for _ in range(len(medoids)):
        clusters.append([])

    cost = 0
    # razdeli instance v skupine (clusterje)
    for i in range(len(data)):
        d_min = 100000
        cluster = -1
        # za vsako instanco izračunamo oddaljenost od vsake medoide
        for j in range(len(medoids)):
            d = cosine_dist(list(data.values())[i], data[medoids[j]])
            d = np.sum(d)
            # instanco dodamo v medoido z najmanjšo razdaljo
            if(d < d_min):
                d_min = d
                cluster = j
        
        clusters[cluster].append(list(data.keys())[i])
        cost += d_min
    return cost, clusters

def get_new_medoids(data, clusters):
    # inicializiramo začetni array novih medoid
    medoids = ["" for x in range(len(clusters))]

    for i in range(len(clusters)):
        dist_min = 100000
        # za vsako instanco v vsakem clusterju izračunamo oddaljenost 
        # do ostalih instanc v tem clusterju
        for img in clusters[i]:
            dist = 0
            for k in clusters[i]:
                dist += cosine_dist(data[img], data[k])
            # če je oddaljenost trenutne instance manjša od oddaljenosti prejšnjih instanc
            # to trenutno instanco vzamemo kot potencialno medoido
            if dist < dist_min:
                dist_min = dist
                medoids[i] = img
    
    return medoids

def read_data(path):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=True)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embeddings = {}
    for subdirs, dirs, files in os.walk(path):
        for file in files:
            f = os.path.join(subdirs, file)
            img = Image.open(f)
            input_tensor = preprocess(img)
            input_batch = input_tensor.unsqueeze(0)
        
            with torch.no_grad():
                output = model(input_batch)
            
            embeddings[file] = output[0]

    return embeddings

def cosine_dist(d1, d2):
    cos = 1 - np.dot(d1, d2)/(np.linalg.norm(d1)*np.linalg.norm(d2))
    return cos

def k_medoids(data, medoids):
    # ta (čudna) inicializacija je potrebna zaradi prvih dveh iteracij v while loopu
    cost = 99999
    cost_prev = 100000
    while(cost < cost_prev):
        cost_prev = cost

        # za vsako instanco izračunamo cost za ta cluster
        cost, clusters = get_clusters_and_cost(data, medoids)

        # za vsak cluster, preračunamo novo medoido
        medoids = get_new_medoids(data, clusters)

    return clusters

def silhouette(el, clusters, data):
    a = 0
    b = 0
    b_min = 10000

    # iteriramo čez vsak cluster
    # posebej obravnavamo cluster, v katerem se nahaja naš element, posebej pa vsej ostale
    for i in clusters:
        b = 0
        # cluster, v katerem je naš element
        if el in i:
            # za vsako instanco tega clusterja (razen naš element!) 
            # izračunamo (kumulativno) cos_dist
            for img in i:
                if(img != el):
                    a += cosine_dist(data[img], data[el])
            try:
                a = a/(len(i)-1)
            except:
                return 0

        else:
            for img in i:
                b += cosine_dist(data[img], data[el])
            if(b < b_min):
                b = b/len(i)
                b_min = b

    s = (b_min - a)/max(a, b_min)
    return s

def silhouette_average(data, clusters):
    avg_silueta = 0
    for i in clusters:
        for k in i:
            avg_silueta += silhouette(k, clusters, data)
    
    return avg_silueta/len(data)

def get_img(I, path):
    img_type = (I[:-4]).capitalize()
    img_type = re.sub(r'[0-9]', '', img_type)
    img_path = os.path.join(path, img_type)
    img = plt.imread(os.path.join(img_path, I))
    
    return img

def main():
    if len(sys.argv) == 3:
        K = sys.argv[1]
        path = sys.argv[2]
    else:
        K = 5
        path = "./Pets"

    K = int(K) # pretvorba iz int -> str
    random.seed(10) 
    data = read_data(path)
    
    # 100 iteracij z naključnim začetnim izborom medoidov
    # na podlagi največje silhuete, izberemo najboljše začetne medoide
    best_clusters = []
    avg_sil_best = 0
    avg_silhueta = 0
    for _ in range(100):
        medoids = random.sample(list(data), int(K))
        clusters = k_medoids(data, medoids)
        avg_silhueta = silhouette_average(data, clusters)
        if avg_silhueta > avg_sil_best:
            avg_sil_best= avg_silhueta
            best_clusters = clusters

    c_plt = 0
    for cluster in range(len(best_clusters)):
        silhuete = []

        for i in best_clusters[cluster]:
            img = get_img(i, path)
            silhueta = silhouette(i, best_clusters, data)
            silhuete.append((np.round(silhueta, 3), img))
        
        silhuete.sort(key= lambda x: x[0], reverse=True)
        counter = 0
        for i in silhuete:
            counter += 1
            plt.subplot(1, len(silhuete), counter)
            plt.xticks([])
            plt.yticks([])
            plt.title(i[0], fontdict={'fontsize': 4})
            plt.imshow(i[1])
            
        plt.savefig('./plt' + str(c_plt), dpi=150)
        c_plt+=1

if __name__ == "__main__":
    # import brez izvajanja!
    main()
    